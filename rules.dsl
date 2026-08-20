    runtime iomp {
      // How the global thread id is acquired
      global_tid_function = "__kmpc_global_thread_num";

      // Chunk size for the static work-sharing schedule; flows into the final
      // `chunk` argument of __kmpc_for_static_init_4 (wsloop construct).
      let default_chunk = 1;

      construct parallel {
        // ABI selector.  by_pointer entails the Intel microtask signature
        // void(ptr gtid, ptr btid, cap0, cap1, ...), built in C++ (see
        // outlineConstruct): each capture is passed as its own trailing arg.
        capture_strategy = by_pointer;

        pre {
          // No `emit` for ident or global_tid: both are materialised on demand
          // from the references below, exactly like barrier/wsloop/task resolve
          // their `ident`/`global_tid` args.  ident is always needed (the fork);
          // global_tid only when one of these two optional push calls is present
          // — the pass derives the need from usage, no DSL declaration.  The
          // gtid function name comes from the `global_tid_function` property.
          when has(num_threads) => call "__kmpc_push_num_threads"(ident, global_tid, num_threads);
          when has(proc_bind) => call "__kmpc_push_proc_bind"(ident, global_tid, proc_bind);
        }

        invoke {
          // if(cond) branches on a runtime value, so it is a `branch` and not a
          // `when`.  With no if clause the condition is null and only the fork
          // survives.  __kmpc_fork_call_if is deliberately not used: it takes a
          // single packed void* (argc <= 1), while by_pointer passes each
          // capture as its own vararg.
          branch if_clause {
            true  => call "__kmpc_fork_call"(ident, argc(captures), body, captures);
            // Serialized: run the region on this thread between the runtime's
            // begin/end pair.  The microtask ABI takes gtid and btid by
            // pointer, so the two slots come from the outlining pass; the
            // callee is `body` itself, a bound value rather than a name.
            false => {
              call "__kmpc_serialized_parallel"(ident, global_tid);
              call body(gtid_addr, btid_addr, captures);
              call "__kmpc_end_serialized_parallel"(ident, global_tid);
            }
          }
        }
      }

      construct barrier {
        invoke {
          call "__kmpc_barrier"(ident(barrier_expl), global_tid);
        }
      }

      construct taskwait {
        // Wait on the completion of the current task's child tasks.  Like
        // barrier: no body, no captures, a single call using ident + gtid.
        // depend/nowait clauses are ignored in v1.
        invoke {
          call "__kmpc_omp_taskwait"(ident, global_tid);
        }
      }

      construct wsloop when schedule == static {
        pre {
          call "__kmpc_for_static_init_4"(ident(work_loop), global_tid, 34, last, lower, upper, stride, step, default_chunk);
        }
        invoke {
          emit loop_body;
        }
        post {
          call "__kmpc_for_static_fini"(ident(work_loop), global_tid);
          when not nowait => call "__kmpc_barrier"(ident(barrier_impl_for), global_tid);
        }
      }

      construct task {
        // ABI selector.  shareds entails the Intel task-routine signature
        // i32(i32 gtid, ptr task): captures live in a runtime-allocated shareds
        // struct reached via load(task->shareds), emitted via the
        // __kmpc_omp_task_alloc/task two-call sequence (see outlineTaskEntry).
        capture_strategy = shareds;
        // kmp_task_t header ABI layout, DSL-owned (like task_flags below).
        // Consumed by the pass (as capture_strategy is): field 0 is the
        // shareds pointer the entry prolog loads, and sizeof(this) is the
        // `task_size` passed to __kmpc_omp_task_alloc.
        kmp_task_t = struct(ptr, ptr, i32, ptr, ptr);
        // Task allocation flags for __kmpc_omp_task_alloc.  1 = tied; the
        // value lives here (like `default_chunk`) so the runtime ABI constant
        // is DSL-owned rather than hardcoded in the pass.  final TBD: it would
        // OR an extra bit in via a `when has(...)` chain.  `if` needs no flag
        // bit — it is a call-site branch, see the invoke block below.
        let task_flags = 1;
        // No pre block: unlike parallel every task invoke call uses
        // both ident and global_tid, so there is no optionality.
        invoke {
          // `let task = call ...` binds the __kmpc_omp_task_alloc result, which
          // every call below takes; `populate_shareds(task)` is a C++-backed
          // verb that writes the captures into task->shareds (more documented
          // in docs/lowering-specs/task-lowering-spec.md).  task_flags is the
          // `let` above.
          let task = call "__kmpc_omp_task_alloc"(ident, global_tid, task_flags,
                                                  task_size, shareds_size, body);
          emit populate_shareds(task);
          // if(cond) branches on a runtime value, so it is a `branch` and not a
          // `when`.  The task is allocated and populated either way — the
          // undeferred path takes a kmp_task_t* too.  With no if clause the
          // condition is null and only the deferred call survives.
          branch if_clause {
            // Deferred: hand the task to the runtime's scheduler.
            true  => call "__kmpc_omp_task"(ident, global_tid, task);
            // Undeferred: run it now on the spawning thread, between the
            // runtime's begin/end pair.  The entry is called through the bound
            // pointer rather than by name, because the outlined function's real
            // name is known only to the outlining pass.
            false => {
              call "__kmpc_omp_task_begin_if0"(ident, global_tid, task);
              call body(global_tid, task);
              call "__kmpc_omp_task_complete_if0"(ident, global_tid, task);
            }
          }
        }
      }
    }


runtime libgomp {
  construct parallel {
    // ABI selector.  packed entails the closure signature void(ptr data): all
    // captures live in one struct the call site hands over by pointer (env_ptr).
    capture_strategy = packed;
    pre {}
    invoke {
      // GOMP_parallel has no `if` parameter: GCC lowers the clause by running
      // the region with a one-thread team, which is the serial execution.  That
      // is a choice on a *runtime* value, so it is a branch and not a `when`.
      // The team size for the true side is still a compile-time choice, hence
      // the nested when/otherwise; 0 there means "runtime default".
      // With no if clause the condition is null, the branch collapses, and only
      // the true arm is emitted.
      //
      // The last argument is GOMP_parallel's flags word, which carries the
      // proc_bind policy in its low bits — same numbering as iomp's
      // kmp_proc_bind_t, and 0 when no policy was asked for.  There is no push
      // call to gate here, so the token used is the always-valued one.
      branch if_clause {
        true => {
          when has(num_threads) => call "GOMP_parallel"(body, env_ptr, num_threads, proc_bind_flags);
          otherwise             => call "GOMP_parallel"(body, env_ptr, 0, proc_bind_flags);
        }
        false => call "GOMP_parallel"(body, env_ptr, 1, proc_bind_flags);
      }
    }
  }
  construct wsloop when schedule == static {
    thread_id_function   = "omp_get_thread_num";
    num_threads_function = "omp_get_num_threads";
    pre {
      emit thread_bounds;
    }
    invoke {
      emit loop_body;
    }
    post {
      when not nowait => call "GOMP_barrier"();
    }
  }
  construct barrier {
    invoke {
      call "GOMP_barrier"();
    }
  }
  construct taskwait {
    // Wait on the current task's child tasks.  Leaf construct like barrier:
    // no body, no captures; GOMP_taskwait takes no arguments.
    invoke {
      call "GOMP_taskwait"();
    }
  }
  construct task {
    // ABI selector.  packed entails the closure signature void(ptr data): all
    // captures live in one struct the call site hands over by pointer (env_ptr).
    capture_strategy = packed;
    invoke {
      when has(if_clause) =>
        call "GOMP_task"(body, env_ptr, null,
                         env_size, env_align,
                         if_clause, 0, null, 0, null);
      otherwise =>
        call "GOMP_task"(body, env_ptr, null,
                         env_size, env_align,
                         true, 0, null, 0, null);
    }
  }
}


runtime pmsis {
  construct parallel {
    // ABI selector.  packed entails the closure signature void(ptr data): all
    // captures live in one struct the call site hands over by pointer (env_ptr).
    capture_strategy = packed;
    // Team size used when no num_threads clause asks for one.  The cluster has
    // 8 cores; the value lives here rather than in the pass for the same reason
    // default_chunk and task_flags do — it is a property of the runtime.
    let default_team_size = 8;
    pre {}
    invoke {
      // ext_pi_cl_team_fork has no `if` parameter, so the clause lowers the way
      // libgomp's does: a one-core team, which is the serial execution.  Forking
      // a team of 1 rather than calling the closure directly keeps the region
      // inside a team, so an ext_pi_cl_team_barrier in the body still meets the
      // number of cores it waits for.  That is a choice on a *runtime* value,
      // hence a branch and not a `when`; with no if clause the condition is
      // null, the branch collapses, and only the true arm is emitted.
      branch if_clause {
        true => {
          when has(num_threads) => call "ext_pi_cl_team_fork"(num_threads, body, env_ptr);
          otherwise             => call "ext_pi_cl_team_fork"(default_team_size, body, env_ptr);
        }
        false => call "ext_pi_cl_team_fork"(1, body, env_ptr);
      }
    }
  }
  construct barrier {
    invoke {
      call "ext_pi_cl_team_barrier"();
    }
  }
  // The guard matters even though this is the only wsloop variant here: the
  // cluster has no dispatch API, so `emit thread_bounds` can only ever produce
  // a static block distribution.  Without it any schedule kind would match and
  // be lowered as static — a wrong answer rather than a rejected one.  The
  // context defaults to static when no schedule clause was written, so a bare
  // `omp.wsloop` still matches.
  construct wsloop when schedule == static {
    thread_id_function   = "ext_pi_core_id";
    num_threads_function = "ext_pi_cl_nb_cores";
    pre {
      emit thread_bounds;
    }
    invoke {
      emit loop_body;
    }
    post {
      when not nowait => call "ext_pi_cl_team_barrier"();
    }
  }
}
