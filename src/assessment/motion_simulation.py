"""Test fixtures only. Estimators must never import motion commands or truth."""
import heapq
import math
import random


class RandomSpin:
    def __init__(self,seed=42,allow_reverse=False,omega_max=6.,alpha_max=10.,tau=.2):
        if any(not math.isfinite(v) or v<=0 for v in (omega_max,alpha_max,tau)) or omega_max<1:
            raise ValueError("invalid motion bounds")
        self.rng=random.Random(seed);self.allow_reverse=allow_reverse
        self.omega_max=omega_max;self.alpha_max=alpha_max;self.tau=tau
        self.theta=0.;self.omega=0.;self.alpha=0.;self.target=0.;self.remaining=0.

    def step(self,dt):
        if not math.isfinite(dt) or not 0<dt<=.05:
            raise ValueError("dt must be in (0, .05]")
        self.remaining-=dt
        if self.remaining<=0:
            self.target=self.rng.uniform(1.,self.omega_max)
            if self.allow_reverse and self.rng.random()<.5:self.target=-self.target
            self.remaining=self.rng.uniform(.3,.8)
        change=(self.target-self.omega)*(-math.expm1(-dt/self.tau))
        change=max(-self.alpha_max*dt,min(self.alpha_max*dt,change))
        old=self.omega;self.omega+=change;self.theta+=(old+self.omega)*dt/2;self.alpha=change/dt
        return self.theta,self.omega


class ObservationDeliveryQueue:
    """Only releases already generated observations at their receipt time."""
    def __init__(self):self.pending=[];self.serial=0

    def push(self,observation,acquired_at,delay_s):
        if not all(math.isfinite(t) for t in (acquired_at,delay_s)) or acquired_at<0 or delay_s<0:
            raise ValueError("invalid simulated delay")
        self.serial+=1
        heapq.heappush(self.pending,(acquired_at+delay_s,self.serial,acquired_at,observation))

    def release(self,now):
        output=[]
        while self.pending and self.pending[0][0]<=now+1e-12:
            received_at,_,acquired_at,observation=heapq.heappop(self.pending)
            output.append((acquired_at,received_at,observation))
        return output
