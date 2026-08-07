# `watchdog` — design notes

A deadline on the host's wait for a launch. Fifty lines of code, and four
decisions that are not obvious from them.

## The failure it is for

A CUDA launch is asynchronous and a CUDA wait is not interruptible. Every
`cuStreamSynchronize` and `cuEventSynchronize` in the driver blocks until the
work in front of it retires, with no timeout parameter — so a kernel that stops
making progress stops the *host* with it. The process is alive, holding a
context, printing nothing, and the only thing that will ever end it is something
outside the process.

That has cost real money here twice, in two different shapes:

- **#146**, `bench --case sol-k` at `4096x4096x1024`: the item handoff was one
  deep against a chain of back-pressure that needs four, one warp waited on a
  parity that would never flip, and the container printed nothing for 1200 s
  until `scripts/modal-run`'s silence budget stopped it. Three issues (#146,
  #148, #149) and a container each went into bracketing where the boundary was,
  because a launch that does not return takes the position of all six warps with
  it.
- **the batching exclusion**: `bench --case sol-k` and `profile` were then kept
  out of `session --arms` for the reason above — an arm that can wedge takes
  every row after it — which is a permanent tax on the cheapest lever this repo
  has for GPU spend (CI.md, spending policy rule 2).

`sol_watch.rs` answered the first shape from the *device* side: the same body
with `Semaphore::wait_before` on every spin, so the launch terminates carrying
where each warp was. That is the better instrument when you already suspect a
particular kernel and can afford to rebuild it with marks in it. It is not a
guard: it covers one device body, it perturbs the race it measures, and nothing
about it helps a launch nobody has written a watched variant of.

This module is the host-side complement, and it is the general one: it makes no
claim about *why* a launch did not return, only that it did not, and it applies
to every launch in the tree at once.

## Poll rather than block, and spin before sleeping

The whole mechanism is that `cuEventQuery` exists: it answers "has this event
retired" without blocking, so a wait can be a loop the host controls the clock
of. `wait` records an untimed event behind whatever is queued on the stream and
polls it; `wait_event` polls one the caller already has.

The loop spins for the first 50 ms and sleeps 1 ms a poll after that, and the
spin is the part that matters. `cuEventSynchronize` under `CU_CTX_SCHED_AUTO`
— which is what these runs get — spins too, so for any launch of a normal length
this is the *same wait*, not an addition to it. That is the property
`examples/src/bench.rs` needs in its timed loop, where the guarded wait sits
between `stop.record` and `elapsed_ms` reading a duration off that same pair: no
event is created, no allocation is made, and the bracket around the launch is
untouched. Past 50 ms there is nothing left to be prompt for and a wedge should
not hold a core for the rest of its budget, so the poll goes to sleep.

## The deadline is fatal, and it aborts

Returning an error would be tidier and would be a lie. When the deadline
expires, the launch is still running: the stream still has work on it, the
buffers the row allocated are still owned by a context that will not release
them, and any device the caller went on to time would be a device still busy
with the launch it gave up on. There is no recovery inside the process, so the
API does not offer one — `expired` prints and does not return.

It is `abort()` and not `exit()`. `exit` runs the C runtime's `atexit`
handlers, and the CUDA driver installs one that wants the context — the thing
that is wedged. Aborting runs none of them; the kernel dies when the operating
system reclaims the process, which is the only reclamation available. The cost
is that the exit status is `SIGABRT`, which `subprocess` reports as `-6`;
`modal_app.py` names that number in its session summary so it reads as a
deadline rather than as a mystery.

The isolation the design needs is therefore a *process* boundary, and it already
existed: `modal_app.py`'s `_run` is a subprocess, so one arm of a session is one
process. What had to change was the session giving up when one of them failed.

## 30 s, and why it is not a multiple of a launch

The default budget is 30 s. That is not `measured × 10`, and the arithmetic says
why it should not be: the slowest single launch anywhere in this tree is the
16384³ GEMM at about 8 ms, and ten times 8 ms is a deadline that would fire on
a driver hiccup. A guarded wait also covers the staging copies queued ahead of
it and the driver's lazy load of a module on its first launch, which are seconds
rather than milliseconds and are not measured anywhere.

So the budget is chosen from both ends instead: three orders of magnitude above
the work it is waiting for, and two below the 3600 s ceiling a wedge used to
ride at B200 rates. `KITTENS_LAUNCH_BUDGET_MS` moves it for one process, and a
value that is not a positive number is the default rather than the absence of a
deadline — a budget switched off by a typo is the failure mode worth refusing.

## The label is the announcement

A deadline that fires is only useful if it says what it fired on, and the sweeps
were already printing that: `bench.rs::announce` is the row announcement, and
`sol::sweep_k`'s doc has said since #149 that *the last announced row is the one
that did not return*. `watching` is that same string, kept where the watchdog
can print it — so the sweep says where it stopped even when nobody was reading
the stream it said it on. A sweep that announces nothing falls back to its own
command line, which still names the case.

## Coverage, stated honestly

Guarded: every `stream.synchronize()` in the three kernel crates, and every
readback, through `ReadBack::read_back` — `DeviceBuffer::to_host_vec`
synchronizes inside itself, so an unguarded readback after a launch is an
unguarded wait, and every correctness check in this tree is one.

Not guarded, by design: a wedge with no launch in it. A container that never
reaches Python, a `cargo` that hangs, a Modal-side eviction — those are
`scripts/modal-run`'s startup and silence budgets, one level out, and CI.md
describes them. Three levels, each bounding what the one below cannot see: a
launch (30 s, this module), a container's silence (300/1200 s, the wrapper), and
the function itself (`timeout=`, Modal).

## The control

`modal_app.py::wedge_demo`, and `experiments/src/wedge.rs` behind an
off-by-default feature: a kernel that spins for two minutes against a five-second
budget, queued **immediately in front of a row's own launch, on that row's own
stream** — everything loaded, everything staged, then a launch that will not
finish. A three-arm session goes through it and must come back with exactly the
middle arm failed.

That exists for the reason `modal_app.py::stall` exists one level out: a check
nobody has watched fail is not a check. The spin is bounded rather than infinite
so that a demonstration which goes wrong still gives the device back — which has
been worth having.

**It earned its keep before it ever passed.** The first attempt injected the spin
at the top of the process rather than in front of a row, on the theory that
wedging earlier can only be a stronger test. It sat in `DeviceBuffer::from_host`,
which is why `stage` and `cleared` exist at all. The second, with those guarded,
stopped before the sweep had printed its own header — ahead of the first guarded
wait, with `kernels::load` the only candidate, which is a driver JIT this tree
cannot put an event behind. Neither is the failure the watchdog is for, and a
process wedged before it has loaded its kernels is a different test rather than a
stronger one. The injection point moved to where #146 happened, and the spin
gained a store in its loop so that "bounded" is a property rather than a hope.
