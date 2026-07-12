    runtime iomp {
      // How the global thread id is acquired
      global_tid_function = "__kmpc_global_thread_num";

      // Chunk size for the static work-sharing schedule; flows into the final
      // `chunk` argument of __kmpc_for_static_init_4 (wsloop construct).
      let default_chunk = 1;

      construct parallel {
        // ABI tag, matched head-only (.find("microtask")).  The args
        // (ptr_tid, ptr_btid, captures) were decorative so they are dropped;
        // the actual microtask shape is built in C++ (see outlineConstruct).
        outline_signature = microtask();
        capture_strategy = "by_pointer";

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
          when has(if_clause) => call "__kmpc_fork_call_if"(ident, argc(captures), body, if_clause, captures);
          otherwise => call "__kmpc_fork_call"(ident, argc(captures), body, captures);
        }
      }

      construct barrier {
        invoke {
          call "__kmpc_barrier"(ident(barrier_expl), global_tid);
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
        // No capture_strategy: the task has a single valid topology 
        // (the same "packed" struct as the closure path,
        // runtime-allocated and reached via load(task->shareds)), entailed by
        // the ABI.  The signature is an ABI
        // tag matched by .find("task_entry").
        outline_signature = task_entry();
        // kmp_task_t header ABI layout, DSL-owned (like task_flags below).
        // Consumed by the pass (as outline_signature is): field 0 is the
        // shareds pointer the entry prolog loads, and sizeof(this) is the
        // `task_size` passed to __kmpc_omp_task_alloc.
        kmp_task_t = struct(ptr, ptr, i32, ptr, ptr);
        // Task allocation flags for __kmpc_omp_task_alloc.  1 = tied; the
        // value lives here (like `default_chunk`) so the runtime ABI constant
        // is DSL-owned rather than hardcoded in the pass.  if/final TBD: those
        // would OR extra bits in via a `when has(...)` chain.
        let task_flags = 1;
        // No pre block: unlike parallel every task invoke call uses
        // both ident and global_tid, so there is no optionality.
        invoke {
          // `let task = call ...` binds the __kmpc_omp_task_alloc result;
          // `populate_shareds(task)` is a C++-backed verb that writes the
          // captures into task->shareds (more documented in
          // docs/task-lowering-spec.md).  task_flags is the `let` above.
          let task = call "__kmpc_omp_task_alloc"(ident, global_tid, task_flags,
                                                  task_size, shareds_size, body);
          emit populate_shareds(task);
          call "__kmpc_omp_task"(ident, global_tid, task);
        }
      }
    }


runtime libgomp {
  construct parallel {
    outline_signature = closure();   // ABI tag, head-only matched; env_ptr is kept (used by the invoke), just not as a decorative signature arg
    capture_strategy = "packed";
    pre {}
    invoke {
      when has(num_threads) => call "GOMP_parallel"(body, env_ptr, num_threads, 0);
      otherwise             => call "GOMP_parallel"(body, env_ptr, 0, 0);
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
  construct task {
    outline_signature = closure();   // ABI tag, head-only matched; env_ptr is kept (used by the invoke), just not as a decorative signature arg
    capture_strategy  = "packed";
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
    outline_signature = closure();   // ABI tag, head-only matched; env_ptr is kept (used by the invoke), just not as a decorative signature arg
    capture_strategy = "packed";
    pre {}
    invoke {
      call "ext_pi_cl_team_fork"(8, body, env_ptr);
    }
  }
  construct barrier {
    invoke {
      call "ext_pi_cl_team_barrier"();
    }
  }
  construct wsloop {
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
