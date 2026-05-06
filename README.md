"""
Pong – Reinforcement Learning via Background Self-Play
=======================================================
While you play, a pool of worker threads silently simulate thousands of
headless AI-vs-AI games per minute.  Every N simulated games the workers
merge their experience back into the shared brain and mutate slightly,
forming a new generation.  The live AI uses the latest brain.

Architecture
------------
  BrainPool   – thread-safe shared brain + experience queue
  HeadlessGame – pure-Python (no tkinter) pong simulation; runs in threads
  SelfPlayWorker – spawns HeadlessGame matches in a tight loop
  self_update – reinforcement update: reward → param nudges + exploration noise
  Game (tkinter) – the visible game; reads latest params from BrainPool

Changes
-------
  - Pause/resume button for background simulation (⏸/▶ toggle)
  - MERGE_EVERY = 700  (simulate 700 games per generation merge)
  - Unlimited history  (no cap on remembered games)
"""

from tkinter import *
import random, math, json, os, time, copy, threading, queue, multiprocessing

# ─── Persistence ──────────────────────────────────────────────────────────────
BRAIN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_brain.json")

PARAM_LIMITS = {
    "track_speed":     [4.0,  16.0],
    "dodge_threshold": [0.20,  0.85],
    "dodge_speed":     [7.0,  20.0],
    "shoot_alignment": [25,   180],
    "shoot_eagerness": [0.001, 0.030],
    "aggression":      [0.4,   2.0],
}

DEFAULT_PARAMS = {
    "track_speed":     8.0,
    "dodge_threshold": 0.55,
    "dodge_speed":     11.0,
    "shoot_alignment": 80.0,
    "shoot_eagerness": 0.008,
    "aggression":      1.0,
}

DEFAULT_BRAIN = {
    "params":        copy.deepcopy(DEFAULT_PARAMS),
    "best_params":   copy.deepcopy(DEFAULT_PARAMS),
    "best_winrate":  0.0,
    "history":       [],          # unlimited — no cap
    "generation":    0,
    "sim_games":     0,
    "last_thought":  "No games yet.",
}

def clamp(v, lo, hi): return max(lo, min(hi, v))

def load_brain():
    if os.path.exists(BRAIN_FILE):
        try:
            b = json.load(open(BRAIN_FILE))
            for k,v in DEFAULT_BRAIN.items():
                b.setdefault(k, copy.deepcopy(v))
            for k,v in DEFAULT_PARAMS.items():
                b["params"].setdefault(k, v)
                b["best_params"].setdefault(k, v)
            return b
        except Exception:
            pass
    return copy.deepcopy(DEFAULT_BRAIN)

def save_brain(brain):
    b = copy.deepcopy(brain)
    # No history truncation — remember every game
    with open(BRAIN_FILE, "w") as f:
        json.dump(b, f, indent=2)


# ─── Reinforcement update ─────────────────────────────────────────────────────
def self_update(params, stats, lr_scale=1.0):
    """
    Given outcome stats for ONE game, return updated params.
    Works for both headless self-play and real player games.
    """
    p = copy.deepcopy(params)
    s = stats

    reward = (s["ai_score"] - s["player_score"]
              + 0.5 * s["shots_hit"]
              - 0.5 * s["times_hit_by_shot"]
              + 0.1 * s["times_dodged"])

    won  = s["ai_score"] >= 5
    mag  = min(abs(reward) / 5.0, 1.0)
    lr   = 0.04 * mag * lr_scale
    thoughts = []

    # track_speed
    ball_goals_against = max(0, s["player_score"] - s["times_hit_by_shot"])
    if ball_goals_against > s["ai_score"]:
        p["track_speed"] = clamp(p["track_speed"] + lr*0.6, *PARAM_LIMITS["track_speed"])
        thoughts.append("speed↑")
    elif won and ball_goals_against == 0:
        p["track_speed"] = clamp(p["track_speed"] - lr*0.1, *PARAM_LIMITS["track_speed"])

    # dodge
    total_dodge_events = s["times_hit_by_shot"] + s["times_dodged"]
    hit_rate = s["times_hit_by_shot"] / max(1, total_dodge_events)
    if hit_rate > 0.3:
        p["dodge_threshold"] = clamp(p["dodge_threshold"] - lr*0.5, *PARAM_LIMITS["dodge_threshold"])
        p["dodge_speed"]     = clamp(p["dodge_speed"]     + lr*2.0, *PARAM_LIMITS["dodge_speed"])
        thoughts.append("dodge↑")
    elif hit_rate == 0 and s["times_dodged"] > 2:
        p["dodge_threshold"] = clamp(p["dodge_threshold"] + lr*0.1, *PARAM_LIMITS["dodge_threshold"])

    # shooting
    accuracy = s["shots_hit"] / max(1, s["shots_fired"])
    if s["shots_fired"] == 0:
        p["shoot_eagerness"] = clamp(p["shoot_eagerness"]*1.15, *PARAM_LIMITS["shoot_eagerness"])
        thoughts.append("shoot↑(never fired)")
    elif accuracy < 0.15:
        p["shoot_alignment"] = clamp(p["shoot_alignment"] - lr*10, *PARAM_LIMITS["shoot_alignment"])
        p["shoot_eagerness"] = clamp(p["shoot_eagerness"]*0.90,    *PARAM_LIMITS["shoot_eagerness"])
        thoughts.append("aim↑")
    elif accuracy > 0.45:
        p["shoot_eagerness"] = clamp(p["shoot_eagerness"]*1.10,    *PARAM_LIMITS["shoot_eagerness"])
        p["shoot_alignment"] = clamp(p["shoot_alignment"] + lr*5,  *PARAM_LIMITS["shoot_alignment"])
        thoughts.append("shoot more")

    # aggression
    if won:
        p["aggression"] = clamp(p["aggression"]*0.97, *PARAM_LIMITS["aggression"])
    else:
        p["aggression"] = clamp(p["aggression"]*1.06, *PARAM_LIMITS["aggression"])
        thoughts.append("aggr↑")

    # exploration noise (always mutate slightly)
    noise = 0.012
    for key,(lo,hi) in PARAM_LIMITS.items():
        p[key] = clamp(p[key] + random.gauss(0, noise*(hi-lo)), lo, hi)

    thought_str = ", ".join(thoughts) if thoughts else "minor mutations"
    return p, thought_str


# ─── Headless Pong simulation (no GUI) ────────────────────────────────────────
class HeadlessBall:
    W, H = 800, 600

    def __init__(self):
        self.reset()

    def reset(self):
        self.x  = self.W / 2
        self.y  = self.H / 2
        d = random.choice([-1,1])
        a = random.uniform(-30,30)
        spd = 6.0
        self.base_speed = self.current_speed = spd
        ar = math.radians(a)
        self.vx = d * spd * math.cos(ar)
        self.vy =     spd * math.sin(ar)

    def step(self):
        mag = math.sqrt(self.vx**2 + self.vy**2)
        if mag > 0:
            self.vx *= self.current_speed / mag
            self.vy *= self.current_speed / mag
        self.x += self.vx
        self.y += self.vy
        if self.y <= 0 or self.y >= self.H:
            self.vy *= -1
            self.y = clamp(self.y, 0, self.H)
        if self.x <= 0:  return 'right_scores'
        if self.x >= self.W: return 'left_scores'
        return None

    def hit_paddle(self, px, py, pw=20, ph=100):
        if (self.x+10 >= px and self.x-10 <= px+pw and
                self.y+10 >= py and self.y-10 <= py+ph):
            off = (self.y - (py+ph/2)) / (ph/2)
            self.current_speed = min(self.current_speed*1.1, self.base_speed*2.5)
            ar = math.radians(off*45)
            d  = 1 if self.vx < 0 else -1
            self.vx = d * self.current_speed * math.cos(ar)
            self.vy =     self.current_speed * math.sin(ar)
            return True
        return False


class HeadlessAI:
    """One AI paddle in a headless game."""
    W, H = HeadlessBall.W, HeadlessBall.H
    SHOOT_CD = 90

    def __init__(self, side, params):
        self.side   = side
        self.params = params
        pw, ph = 20, 100
        self.px = 30 if side=='left' else self.W-50
        self.py = (self.H-ph)//2
        self.pw, self.ph = pw, ph
        self.cd  = 0

    def decide(self, ball, bullets):
        p   = self.params
        cy  = self.py + self.ph/2
        spd = p["track_speed"] * p["aggression"]

        for b in bullets:
            if b["origin"] != self.side:
                bx, by = b["x"], b["y"]
                heading_here = (self.side=='right' and b["vx"]>0) or \
                               (self.side=='left'  and b["vx"]<0)
                if heading_here:
                    thresh = self.W * p["dodge_threshold"]
                    in_range = (bx > thresh if self.side=='right'
                                else bx < self.W-thresh)
                    if in_range:
                        dspd = p["dodge_speed"] * p["aggression"]
                        target = cy+70 if by < cy else cy-70
                        diff   = target - cy
                        dy = dspd if diff>5 else (-dspd if diff<-5 else 0)
                        return dy, False

        diff = ball.y - cy
        dy   = spd if diff>10 else (-spd if diff<-10 else 0)
        shoot = False
        if self.cd == 0 and random.random() < p["shoot_eagerness"]*p["aggression"]:
            shoot = True

        return dy, shoot

    def move(self, dy):
        self.py = clamp(self.py+dy, 0, self.H-self.ph)
        if self.cd > 0: self.cd -= 1

    def center_y(self): return self.py + self.ph/2


def run_headless_game(left_params, right_params, win_score=5):
    ball     = HeadlessBall()
    left_ai  = HeadlessAI('left',  left_params)
    right_ai = HeadlessAI('right', right_params)

    ls = {"ai_score":0,"player_score":0,"shots_fired":0,"shots_hit":0,
          "times_dodged":0,"times_hit_by_shot":0}
    rs = {"ai_score":0,"player_score":0,"shots_fired":0,"shots_hit":0,
          "times_dodged":0,"times_hit_by_shot":0}

    bullets = []
    BSPEED  = 14

    for _ in range(5000):
        ldy, lshoot = left_ai.decide(ball, bullets)
        rdy, rshoot = right_ai.decide(ball, bullets)

        l_dodging = ldy!=0 and any(b["origin"]=='right' for b in bullets)
        r_dodging = rdy!=0 and any(b["origin"]=='left'  for b in bullets)
        if l_dodging: ls["times_dodged"] += 1
        if r_dodging: rs["times_dodged"] += 1

        left_ai.move(ldy); right_ai.move(rdy)

        if lshoot and left_ai.cd==0:
            if abs(left_ai.center_y()-right_ai.center_y()) < left_params["shoot_alignment"]:
                bullets.append({"x":left_ai.px+left_ai.pw, "y":left_ai.center_y(),
                                 "vx":BSPEED, "origin":"left"})
                left_ai.cd = HeadlessAI.SHOOT_CD
                ls["shots_fired"] += 1

        if rshoot and right_ai.cd==0:
            if abs(right_ai.center_y()-left_ai.center_y()) < right_params["shoot_alignment"]:
                bullets.append({"x":right_ai.px, "y":right_ai.center_y(),
                                 "vx":-BSPEED, "origin":"right"})
                right_ai.cd = HeadlessAI.SHOOT_CD
                rs["shots_fired"] += 1

        for b in bullets:
            b["x"] += b["vx"]

        hit_bullets = []
        for b in bullets:
            if b["origin"]=="left":
                rp = right_ai
                if (b["x"]+6 >= rp.px and b["x"]-6 <= rp.px+rp.pw and
                        b["y"]+6 >= rp.py and b["y"]-6 <= rp.py+rp.ph):
                    ls["ai_score"]         += 1
                    rs["player_score"]     += 1
                    rs["times_hit_by_shot"]+= 1
                    ls["shots_hit"]        += 1
                    ball.reset(); hit_bullets.append(b)
            else:
                lp = left_ai
                if (b["x"]+6 >= lp.px and b["x"]-6 <= lp.px+lp.pw and
                        b["y"]+6 >= lp.py and b["y"]-6 <= lp.py+lp.ph):
                    rs["ai_score"]         += 1
                    ls["player_score"]     += 1
                    ls["times_hit_by_shot"]+= 1
                    rs["shots_hit"]        += 1
                    ball.reset(); hit_bullets.append(b)

        bullets = [b for b in bullets if b not in hit_bullets
                   and 0 <= b["x"] <= HeadlessBall.W]

        ball.hit_paddle(left_ai.px,  left_ai.py)
        ball.hit_paddle(right_ai.px, right_ai.py)

        result = ball.step()
        if result=='right_scores':
            ls["ai_score"]    += 1; rs["player_score"] += 1; ball.reset()
        elif result=='left_scores':
            rs["ai_score"]    += 1; ls["player_score"] += 1; ball.reset()

        if ls["ai_score"]>=win_score or rs["ai_score"]>=win_score:
            break

    return ls, rs, ls["ai_score"], rs["ai_score"]


# ─── Thread-safe brain pool ────────────────────────────────────────────────────
class BrainPool:
    MERGE_EVERY   = 700    # simulate 700 games per generation merge
    NUM_WORKERS   = max(2, multiprocessing.cpu_count() - 1)

    def __init__(self):
        self._lock    = threading.Lock()
        self.brain    = load_brain()
        self._results = queue.Queue()
        self._pending = 0
        self._workers = []
        self._running = False
        self._paused  = False          # ← pause flag
        self.status   = f"Workers: {self.NUM_WORKERS}  |  Sim games: {self.brain['sim_games']}"

    # ── Pause / resume ────────────────────────────────────────────────────────
    @property
    def paused(self):
        return self._paused

    def pause(self):
        self._paused = True
        self._update_status_paused()

    def resume(self):
        self._paused = False

    def toggle_pause(self):
        if self._paused:
            self.resume()
        else:
            self.pause()

    def _update_status_paused(self):
        self.status = (
            f"⏸ PAUSED  |  Gen {self.brain['generation']}  |  "
            f"{self.brain['sim_games']:,} sim games"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def start(self):
        self._running = True
        for i in range(self.NUM_WORKERS):
            t = threading.Thread(target=self._worker_loop, name=f"SP-{i}", daemon=True)
            t.start()
            self._workers.append(t)
        t = threading.Thread(target=self._merger_loop, name="Merger", daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def get_params(self):
        with self._lock:
            p = copy.deepcopy(self.brain["params"])
        for key,(lo,hi) in PARAM_LIMITS.items():
            p[key] = clamp(p[key] + random.gauss(0, 0.02*(hi-lo)), lo, hi)
        return p

    def get_best_params(self):
        with self._lock:
            return copy.deepcopy(self.brain["params"])

    def push_result(self, winner_params, loser_params, winner_stats, loser_stats):
        self._results.put((winner_params, loser_params, winner_stats, loser_stats))

    def _worker_loop(self):
        while self._running:
            if self._paused:
                time.sleep(0.2)   # idle while paused
                continue
            lp = self.get_params()
            rp = self.get_params()
            try:
                ls, rs, lscore, rscore = run_headless_game(lp, rp)
                if lscore > rscore:
                    self.push_result(lp, rp, ls, rs)
                else:
                    self.push_result(rp, lp, rs, ls)
            except Exception:
                pass

    def _merger_loop(self):
        batch = []
        while self._running:
            try:
                item = self._results.get(timeout=0.2)
                batch.append(item)
            except queue.Empty:
                pass

            if len(batch) >= self.MERGE_EVERY:
                self._merge(batch)
                batch = []

    def _merge(self, batch):
        with self._lock:
            p = self.brain["params"]
            wins_this_batch = 0

            for winner_p, loser_p, w_stats, l_stats in batch:
                updated_p, thought = self_update(p, w_stats, lr_scale=0.3)
                p = updated_p
                wins_this_batch += 1

            self.brain["sim_games"] += len(batch)
            self.brain["params"]     = p

            sim_wr = wins_this_batch / max(1, len(batch))
            if sim_wr >= self.brain.get("best_winrate", 0):
                self.brain["best_winrate"] = sim_wr
                self.brain["best_params"]  = copy.deepcopy(p)

            self.brain["generation"] += 1
            self.brain["last_thought"] = (
                f"Gen {self.brain['generation']} | "
                f"{self.brain['sim_games']} sim games | "
                f"{thought}"
            )

            self.status = (
                f"Gen {self.brain['generation']}  |  "
                f"{self.brain['sim_games']:,} sim games  |  "
                f"Workers: {self.NUM_WORKERS}"
            )
            save_brain(self.brain)

    def incorporate_real_game(self, stats):
        """Called after a real player game — weighted update."""
        with self._lock:
            updated_p, thought = self_update(self.brain["params"], stats, lr_scale=2.0)
            self.brain["params"] = updated_p
            self.brain["history"].append({
                **stats,
                "gen": self.brain["generation"],
                "ts":  time.strftime("%H:%M:%S"),
                "real": True,
            })
            self.brain["last_thought"] = f"Real game: {thought}"
            save_brain(self.brain)


# ─────────────────────────────────────────────────────────────────────────────
# TKINTER GAME
# ─────────────────────────────────────────────────────────────────────────────

class Projectile:
    SPEED=14; RADIUS=6
    def __init__(self, canvas, origin_side, paddle_pos):
        self.canvas=canvas; self.origin_side=origin_side
        r=self.RADIUS
        cx=(paddle_pos[0]+paddle_pos[2])/2; cy=(paddle_pos[1]+paddle_pos[3])/2
        self.id=canvas.create_oval(cx-r,cy-r,cx+r,cy+r,fill='#FFD700',outline='#FFA500',width=2)
        self.vx=self.SPEED if origin_side=='left' else -self.SPEED
        self.active=True
    def move(self):    self.canvas.move(self.id,self.vx,0)
    def pos(self):     return self.canvas.coords(self.id)
    def destroy(self): self.canvas.delete(self.id); self.active=False
    def hits(self,pp):
        p=self.pos()
        if not p: return False
        return p[2]>=pp[0] and p[0]<=pp[2] and p[3]>=pp[1] and p[1]<=pp[3]
    def off_screen(self,w):
        p=self.pos(); return (not p) or p[2]<0 or p[0]>w


class Ball:
    def __init__(self, canvas):
        self.canvas=canvas
        self.id=canvas.create_oval(0,0,20,20,fill='#FF3333',outline='#FF3333')
        self.canvas_height=canvas.winfo_height(); self.canvas_width=canvas.winfo_width()
        self.reset()
    def reset(self):
        w,h=self.canvas.winfo_width(),self.canvas.winfo_height()
        self.canvas.coords(self.id,w//2-10,h//2-10,w//2+10,h//2+10)
        d=random.choice([-1,1]); a=random.uniform(-30,30); spd=6.0
        self.base_speed=self.current_speed=spd; ar=math.radians(a)
        self.x=d*spd*math.cos(ar); self.y=spd*math.sin(ar)
    def hit_paddle(self,paddle):
        bp=self.canvas.coords(self.id); pp=self.canvas.coords(paddle.id)
        if bp[2]>=pp[0] and bp[0]<=pp[2] and bp[3]>=pp[1] and bp[1]<=pp[3]:
            off=((bp[1]+bp[3])/2-(pp[1]+pp[3])/2)/((pp[3]-pp[1])/2)
            self.current_speed=min(self.current_speed*1.1,self.base_speed*2.5)
            ar=math.radians(off*45); d=1 if self.x<0 else -1
            self.x=d*self.current_speed*math.cos(ar); self.y=self.current_speed*math.sin(ar)
            return True
        return False
    def draw(self):
        mag=math.sqrt(self.x**2+self.y**2)
        if mag>0: self.x*=self.current_speed/mag; self.y*=self.current_speed/mag
        self.canvas.move(self.id,self.x,self.y); pos=self.canvas.coords(self.id)
        if pos[1]<=0 or pos[3]>=self.canvas_height: self.y*=-1
        if pos[0]<=0: return 'right_scores'
        if pos[2]>=self.canvas_width: return 'left_scores'
        return None


class Paddle:
    SHOOT_COOLDOWN=90
    def __init__(self,canvas,color,side,two_player=False,params=None):
        self.canvas=canvas; self.side=side
        self.canvas_width=canvas.winfo_width(); self.canvas_height=canvas.winfo_height()
        self.params=params or copy.deepcopy(DEFAULT_PARAMS)
        pw,ph=20,100
        self.id=canvas.create_rectangle(0,0,pw,ph,fill=color,outline=color)
        x=30 if side=='left' else self.canvas_width-30-pw
        self.canvas.move(self.id,x,(self.canvas_height-ph)//2)
        self.y=0; self.cooldown=0; self._shoot_cb=None
        if side=='left':
            canvas.bind_all('<KeyPress-Up>',    self.move_up)
            canvas.bind_all('<KeyPress-Down>',  self.move_down)
            canvas.bind_all('<KeyRelease-Up>',  self.stop)
            canvas.bind_all('<KeyRelease-Down>',self.stop)
            canvas.bind_all('<KeyPress-space>', self._on_shoot)
        elif side=='right' and two_player:
            canvas.bind_all('<KeyPress-w>',      self.move_up)
            canvas.bind_all('<KeyPress-s>',      self.move_down)
            canvas.bind_all('<KeyRelease-w>',    self.stop)
            canvas.bind_all('<KeyRelease-s>',    self.stop)
            canvas.bind_all('<KeyPress-Return>', self._on_shoot)
    def move_up(self,e):   self.y=-10
    def move_down(self,e): self.y= 10
    def stop(self,e):      self.y=  0
    def _on_shoot(self,e):
        if self._shoot_cb and self.cooldown==0: self._shoot_cb(self)
    def ai_decide(self, ball, projectiles):
        p=self.params; bp=self.canvas.coords(ball.id); pp=self.canvas.coords(self.id)
        cy=(pp[1]+pp[3])/2; spd=p["track_speed"]*p["aggression"]
        for proj in projectiles:
            if not proj.active: continue
            if proj.origin_side!=self.side:
                ppos=proj.pos()
                if not ppos: continue
                px=(ppos[0]+ppos[2])/2; py=(ppos[1]+ppos[3])/2
                heading=(self.side=='right' and proj.vx>0) or (self.side=='left' and proj.vx<0)
                if heading:
                    thresh=self.canvas_width*p["dodge_threshold"]
                    in_r=(px>thresh if self.side=='right' else px<self.canvas_width-thresh)
                    if in_r:
                        dspd=p["dodge_speed"]*p["aggression"]
                        tgt=cy+70 if py<cy else cy-70
                        diff=tgt-cy
                        return (dspd if diff>5 else (-dspd if diff<-5 else 0)), False
        diff=(bp[1]+bp[3])/2-cy
        dy=spd if diff>10 else (-spd if diff<-10 else 0)
        shoot=self.cooldown==0 and random.random()<p["shoot_eagerness"]*p["aggression"]
        return dy, shoot
    def draw(self):
        if self.cooldown>0: self.cooldown-=1
        self.canvas.move(self.id,0,self.y); pos=self.canvas.coords(self.id)
        if pos[1]<=0: self.canvas.move(self.id,0,-pos[1]); self.y=0
        if pos[3]>=self.canvas_height: self.canvas.move(self.id,0,self.canvas_height-pos[3]); self.y=0


class Game:
    def __init__(self, tk, pool):
        self.tk=tk; self.pool=pool
        self.tk.title('Pong – Self-Play RL'); self.tk.wm_attributes('-topmost',1)
        try:    self.tk.state('zoomed')
        except: pass
        try:    self.tk.attributes('-zoomed',True)
        except: pass
        self.tk.update()
        self.canvas=Canvas(self.tk,bg='#1a1a2e',bd=0,highlightthickness=0)
        self.canvas.pack(fill=BOTH,expand=True); self.tk.update()
        self.two_player=False; self.projectiles=[]; self.running=False
        # Persistent canvas IDs for the pause button elements
        self._pause_btn_rect=None; self._pause_btn_text=None
        self.show_mode_select()
        self._tick_status()

    # ── Pause button helpers ──────────────────────────────────────────────────
    def _make_pause_btn(self, cx, cy):
        """Draw the pause/resume toggle button and return (rect_id, text_id)."""
        bw, bh = 160, 38
        bx, by = cx-bw//2, cy-bh//2
        label  = self._pause_label()
        color  = '#553300' if self.pool.paused else '#224422'
        r = self.canvas.create_rectangle(bx,by,bx+bw,by+bh,
                                         fill=color,outline='#AAAAAA',width=1,
                                         tags='pausebtn')
        t = self.canvas.create_text(cx,cy,text=label,
                                    font=('Arial',14,'bold'),fill='white',
                                    tags='pausebtn')
        lc = self._lighten(color)
        for item in (r,t):
            self.canvas.tag_bind(item,'<Button-1>', lambda e: self._on_pause_click())
            self.canvas.tag_bind(item,'<Enter>',
                lambda e,rr=r,l=lc: self.canvas.itemconfig(rr,fill=l))
            self.canvas.tag_bind(item,'<Leave>',
                lambda e,rr=r,oc=color: self.canvas.itemconfig(rr,fill=oc))
        return r, t

    def _pause_label(self):
        return '▶ Resume Sim' if self.pool.paused else '⏸ Pause Sim'

    def _on_pause_click(self):
        self.pool.toggle_pause()
        lbl   = self._pause_label()
        color = '#553300' if self.pool.paused else '#224422'
        if self._pause_btn_rect:
            try:
                self.canvas.itemconfig(self._pause_btn_rect, fill=color)
                self.canvas.itemconfig(self._pause_btn_text, text=lbl)
            except Exception:
                pass

    def _tick_status(self):
        """Update sim-game counter + pause button label every second."""
        if hasattr(self,'_status_var') and self._status_var:
            try: self.canvas.itemconfig(self._status_var, text=self.pool.status)
            except: pass
        # Keep pause button label in sync
        if self._pause_btn_text:
            try:
                self.canvas.itemconfig(self._pause_btn_text, text=self._pause_label())
                color = '#553300' if self.pool.paused else '#224422'
                self.canvas.itemconfig(self._pause_btn_rect, fill=color)
            except: pass
        self.tk.after(1000, self._tick_status)

    # ── MODE SELECT ──────────────────────────────────────────────────────────
    def show_mode_select(self):
        self.running=False; self.canvas.delete('all'); self._unbind_keys()
        self._status_var=None; self._pause_btn_rect=None; self._pause_btn_text=None
        w,h=self.canvas.winfo_width(),self.canvas.winfo_height()
        self.canvas.create_text(w//2,h//2-195,text='⚡ PONG',font=('Arial',72,'bold'),fill='white')
        self.canvas.create_text(w//2,h//2-108,text='Choose game mode',font=('Arial',22),fill='#AAAAAA')
        self._menu_btn(w//2,h//2-28, '🤖  vs Learning AI','#882222',self._start_ai)
        self._menu_btn(w//2,h//2+62, '👥  2 Players',      '#114488',self._start_2p)

        b=self.pool.brain; p=b["params"]
        hi=b["history"]
        real_games=[g for g in hi if g.get("real")]
        real_wins=sum(1 for g in real_games if g.get("ai_score",0)>=5)
        rwr=f"{real_wins/len(real_games)*100:.0f}%" if real_games else "—"

        self.canvas.create_text(w//2,h//2+155,
            text=f"🧠 Gen {b['generation']}  |  {b['sim_games']:,} sim games  |  "
                 f"Real games: {len(real_games)}  win rate: {rwr}",
            font=('Arial',13,'bold'),fill='#7799BB')
        self.canvas.create_text(w//2,h//2+178,
            text=f"speed={p['track_speed']:.1f}  aggr={p['aggression']:.2f}  "
                 f"shoot={p['shoot_eagerness']*100:.2f}%  dodge@{p['dodge_threshold']:.2f}",
            font=('Arial',12),fill='#556688')
        self.canvas.create_text(w//2,h//2+200,
            text=f"💡 {b['last_thought']}",font=('Arial',11,'italic'),fill='#445566')

        # Pause button — bottom-right area
        r, t = self._make_pause_btn(w//2+220, h-55)
        self._pause_btn_rect, self._pause_btn_text = r, t

        self._status_var=self.canvas.create_text(w//2-40,h-45,
            text=self.pool.status,font=('Arial',12),fill='#334466')
        self.canvas.create_text(w//2,h-25,
            text='Left:↑/↓+SPACE  |  Right(2P):W/S+ENTER',
            font=('Arial',13),fill='#333355')

    def _menu_btn(self,cx,cy,label,color,cmd):
        bw,bh=300,65; bx,by=cx-bw//2,cy-bh//2
        r=self.canvas.create_rectangle(bx,by,bx+bw,by+bh,fill=color,outline='white',width=2)
        t=self.canvas.create_text(cx,cy,text=label,font=('Arial',22,'bold'),fill='white')
        lc=self._lighten(color)
        for item in (r,t):
            self.canvas.tag_bind(item,'<Button-1>',lambda e,c=cmd:c())
            self.canvas.tag_bind(item,'<Enter>',   lambda e,rr=r,l=lc: self.canvas.itemconfig(rr,fill=l))
            self.canvas.tag_bind(item,'<Leave>',   lambda e,rr=r,oc=color: self.canvas.itemconfig(rr,fill=oc))

    @staticmethod
    def _lighten(hx):
        h=hx.lstrip('#'); r,g,b=(int(h[i:i+2],16) for i in (0,2,4))
        return '#{:02x}{:02x}{:02x}'.format(min(255,r+55),min(255,g+55),min(255,b+55))

    def _start_ai(self): self.two_player=False; self.start_game()
    def _start_2p(self): self.two_player=True;  self.start_game()

    # ── START GAME ───────────────────────────────────────────────────────────
    def start_game(self):
        self._unbind_keys(); self.canvas.delete('all')
        self.projectiles=[]; self.running=True
        self.left_score=0; self.right_score=0
        self._status_var=None; self._pause_btn_rect=None; self._pause_btn_text=None
        self._stats={"ai_score":0,"player_score":0,"shots_fired":0,
                     "shots_hit":0,"times_dodged":0,"times_hit_by_shot":0}
        self._prev_dodging=False

        params=self.pool.get_best_params()

        w,h=self.canvas.winfo_width(),self.canvas.winfo_height()
        for i in range(0,h,30):
            self.canvas.create_line(w//2,i,w//2,i+15,fill='#333355',width=3)
        gw=10
        self.canvas.create_rectangle(0,0,gw,h,   fill='#2a0a0a',outline='#FF3333',width=3)
        self.canvas.create_rectangle(w-gw,0,w,h,  fill='#0a0a2a',outline='#0066CC',width=3)

        self.left_paddle =Paddle(self.canvas,'#FF3333','left', self.two_player,params)
        self.right_paddle=Paddle(self.canvas,'#0066CC','right',self.two_player,params)
        self.ball=Ball(self.canvas)
        self.left_paddle._shoot_cb =self._fire_left
        self.right_paddle._shoot_cb=self._fire_right

        self.score_text=self.canvas.create_text(w//2,30,
            text=f'{self.left_score}  -  {self.right_score}',
            font=('Arial',36,'bold'),fill='white')

        b=self.pool.brain
        hint=('Left:↑/↓+SPACE  Right:W/S+ENTER' if self.two_player
              else f'↑/↓+SPACE  |  AI gen {b["generation"]}  {b["sim_games"]:,} sim games')
        self.canvas.create_text(w//2,h-50,text=hint,font=('Arial',12),fill='#444466')

        self.left_cd_bar =self.canvas.create_rectangle(20,   h-78,20,   h-66,fill='#FFD700',outline='')
        self.right_cd_bar=self.canvas.create_rectangle(w-20, h-78,w-20, h-66,fill='#FFD700',outline='')

        # Pause button — in-game, top-right corner
        r, t = self._make_pause_btn(w-105, 28)
        self._pause_btn_rect, self._pause_btn_text = r, t

        # Live sim counter
        self._status_var=self.canvas.create_text(w//2,h-28,
            text=self.pool.status,font=('Arial',10,'italic'),fill='#334455')
        self.loop()

    def _fire_left(self,paddle):
        if paddle.cooldown>0: return
        paddle.cooldown=Paddle.SHOOT_COOLDOWN
        self.projectiles.append(Projectile(self.canvas,'left',self.canvas.coords(paddle.id)))

    def _fire_right(self,paddle):
        if paddle.cooldown>0: return
        paddle.cooldown=Paddle.SHOOT_COOLDOWN
        self.projectiles.append(Projectile(self.canvas,'right',self.canvas.coords(paddle.id)))
        self._stats["shots_fired"]+=1

    # ── MAIN LOOP ────────────────────────────────────────────────────────────
    def loop(self):
        if not self.running: return
        w,h=self.canvas.winfo_width(),self.canvas.winfo_height()

        self.left_paddle.draw()

        if not self.two_player:
            move_y,shoot=self.right_paddle.ai_decide(self.ball,self.projectiles)
            if shoot:
                lp=self.canvas.coords(self.left_paddle.id)
                rp=self.canvas.coords(self.right_paddle.id)
                p=self.right_paddle.params
                if abs((lp[1]+lp[3])/2-(rp[1]+rp[3])/2)<p["shoot_alignment"]:
                    self._fire_right(self.right_paddle)
            dodging=(move_y!=0 and any(pr.active and pr.origin_side=='left' for pr in self.projectiles))
            if dodging and not self._prev_dodging: self._stats["times_dodged"]+=1
            self._prev_dodging=dodging
            self.right_paddle.y=move_y

        self.right_paddle.draw()
        self.ball.hit_paddle(self.left_paddle)
        self.ball.hit_paddle(self.right_paddle)
        result=self.ball.draw()
        if result=='right_scores': self.right_score+=1; self._update_score(); self.ball.reset()
        elif result=='left_scores': self.left_score+=1; self._update_score(); self.ball.reset()

        for proj in self.projectiles[:]:
            if not proj.active: continue
            proj.move()
            lp=self.canvas.coords(self.left_paddle.id)
            rp=self.canvas.coords(self.right_paddle.id)
            hit=False
            if proj.origin_side=='left' and proj.hits(rp):
                self.left_score+=1;  self._update_score(); self.ball.reset()
                self._stats["times_hit_by_shot"]+=1; hit=True
            elif proj.origin_side=='right' and proj.hits(lp):
                self.right_score+=1; self._update_score(); self.ball.reset()
                self._stats["shots_hit"]+=1; hit=True
            if hit or proj.off_screen(w):
                proj.destroy(); self.projectiles.remove(proj)

        lc,rc=self.left_paddle.cooldown,self.right_paddle.cooldown
        bmax=120
        lw=int(bmax*(1-lc/Paddle.SHOOT_COOLDOWN)) if lc else bmax
        rw=int(bmax*(1-rc/Paddle.SHOOT_COOLDOWN)) if rc else bmax
        self.canvas.coords(self.left_cd_bar,  20,     h-78,20+lw,  h-66)
        self.canvas.coords(self.right_cd_bar, w-20-rw,h-78,w-20,   h-66)
        self.canvas.itemconfig(self.left_cd_bar,  fill='#FFD700' if lw==bmax else '#886600')
        self.canvas.itemconfig(self.right_cd_bar, fill='#FFD700' if rw==bmax else '#886600')

        if self.left_score>=5:
            self._end_game('YOU WIN!' if not self.two_player else 'LEFT WINS!')
        elif self.right_score>=5:
            self._end_game('AI WINS!' if not self.two_player else 'RIGHT WINS!')
        else:
            self.tk.after(16,self.loop)

    def _update_score(self):
        self.canvas.itemconfig(self.score_text,text=f'{self.left_score}  -  {self.right_score}')

    def _end_game(self,message):
        self.running=False
        self._stats["ai_score"]    =self.right_score
        self._stats["player_score"]=self.left_score
        if not self.two_player:
            self.pool.incorporate_real_game(self._stats)
        self._show_game_over(message)

    def _show_game_over(self,message):
        self._unbind_keys()
        self._pause_btn_rect=None; self._pause_btn_text=None
        w,h=self.canvas.winfo_width(),self.canvas.winfo_height()
        self.canvas.create_rectangle(0,0,w,h,fill='black',stipple='gray50')
        color='#FF3333' if ('YOU' in message or 'LEFT' in message) else '#0066CC'
        self.canvas.create_text(w//2,h//2-135,text=message,font=('Arial',64,'bold'),fill=color)
        self.canvas.create_text(w//2,h//2-55,
            text=f'Final Score: {self.left_score} - {self.right_score}',
            font=('Arial',30),fill='white')

        b=self.pool.brain; p=b["params"]
        self.canvas.create_text(w//2,h//2-12,
            text=f"🧠 Gen {b['generation']}  |  {b['sim_games']:,} sim games",
            font=('Arial',14,'bold'),fill='#7799BB')
        self.canvas.create_text(w//2,h//2+14,
            text=f"speed={p['track_speed']:.1f}  aggr={p['aggression']:.2f}  "
                 f"shoot={p['shoot_eagerness']*100:.2f}%  dodge@{p['dodge_threshold']:.2f}",
            font=('Arial',12),fill='#556688')
        self.canvas.create_text(w//2,h//2+38,
            text=f"💡 {b['last_thought']}",font=('Arial',11,'italic'),fill='#AABB99')

        self._status_var=self.canvas.create_text(w//2,h//2+62,
            text=self.pool.status,font=('Arial',11),fill='#445566')

        # Pause button on game-over screen
        r, t = self._make_pause_btn(w//2, h//2+92)
        self._pause_btn_rect, self._pause_btn_text = r, t

        self._go_btn(w//2,h//2+145,'▶  PLAY AGAIN','#1a5c1a',self.start_game)
        self._go_btn(w//2,h//2+218,'🏠  MAIN MENU', '#2a2a55',self.show_mode_select)

    def _go_btn(self,cx,cy,label,color,cmd):
        bw,bh=250,58; bx,by=cx-bw//2,cy-bh//2
        r=self.canvas.create_rectangle(bx,by,bx+bw,by+bh,fill=color,outline='white',width=2)
        t=self.canvas.create_text(cx,cy,text=label,font=('Arial',21,'bold'),fill='white')
        lc=self._lighten(color)
        for item in (r,t):
            self.canvas.tag_bind(item,'<Button-1>',lambda e,c=cmd:c())
            self.canvas.tag_bind(item,'<Enter>',   lambda e,rr=r,l=lc: self.canvas.itemconfig(rr,fill=l))
            self.canvas.tag_bind(item,'<Leave>',   lambda e,rr=r,oc=color: self.canvas.itemconfig(rr,fill=oc))

    def _unbind_keys(self):
        for k in ('<KeyPress-space>','<KeyPress-Return>',
                  '<KeyPress-Up>','<KeyPress-Down>',
                  '<KeyPress-w>','<KeyPress-s>',
                  '<KeyRelease-Up>','<KeyRelease-Down>',
                  '<KeyRelease-w>','<KeyRelease-s>'):
            try: self.canvas.unbind_all(k)
            except: pass


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pool = BrainPool()
    pool.start()

    root = Tk()
    game = Game(root, pool)

    def on_close():
        pool.stop()
        save_brain(pool.brain)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
