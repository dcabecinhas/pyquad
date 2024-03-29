Names
=====


# From python ideal controller

# Load

state.p = p
state.v = v

# Cable

state.q = q
state.oq = oq

# Quadrotor

state.qR = qR
state.qo = qo

(and also their derivatives)

# Reference trajectory
state.pd = pr(t).reshape(shape(p))
state.dpd = dpr(t).reshape(shape(p))
state.d2pd = d2pr(t).reshape(shape(p))
state.d3pd = d3pr(t).reshape(shape(p))
state.d4pd = d4pr(t).reshape(shape(p))

# Aux vars

state.Fd - Ideal force feedback on the load
state.dFd

state.TLd - Tension on cable
state.TLd_min

state.qd - q_desired_direction
state.dqd = dqd
state.oqd = oqd
state.d2qd
state.doqd

state.mud = qd @ mt(qd) @ TLd    
state.mu = q @ mt(q) @ TLd

dmud
dmu

state.F - Actual force on the load
dF
state.dv - load acceleration
d2v
state.dq = skew(oq) @ q

# Cable Dynamics

(q_actuation, e_q, e_oq) = q_feedback(q,oq,qd,oqd,doqd)

state.q_actuation = q_actuation
state.e_q = e_q
state.e_oq = e_oq

state.de_q

state.a = a
state.u_parallel = u_parallel
state.u_perp = u_perp
state.u = u

# Quadrotor dynamics

state.qRd = qRd
state.qod = qod
state.qtaud = qtaud
state.qtau = qtau
state.dqR = dqR
state.dqo = dqo

qFd = u
r3 = qR @ e3
qT = - mt(r3) @ qFd

state.qFd - Ideal force generated by quad
state.qT - Thrust
state.qF - Real force generated by quad


