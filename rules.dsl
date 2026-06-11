    runtime iomp {
      let default_chunk = 1;

      construct parallel {
        outline_signature = microtask(ptr_tid, ptr_btid, captures);
        capture_strategy = "by_pointer";

        pre {
          emit ident;
          emit global_tid;
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
          call "__kmpc_barrier"(ident, global_tid);
        }
      }

      construct wsloop when schedule == static {
        pre {
          call "__kmpc_for_static_init_4"(ident, global_tid, 34, last, lower, upper, stride, step, 1);
        }
        invoke {
          emit loop_body;
        }
        post {
          call "__kmpc_for_static_fini"(ident, global_tid);
          when not nowait => call "__kmpc_barrier"(ident, global_tid);
        }
      }

      construct wsloop when schedule == dynamic {
        dispatch_init_function = "__kmpc_dispatch_init_4";
        dispatch_next_function = "__kmpc_dispatch_next_4";
        pre {
          emit dispatch_loop;
        }
        invoke {
          emit loop_body;
        }
        post {
          when not nowait => call "__kmpc_barrier"(ident, global_tid);
        }
      }

      // iomp task. Detected by outlineConstruct via outline_signature =
      // task_entry; the alloc -> fill shareds -> submit sequence (with the
      // kmp_task_t ABI: shareds at field 0, i32 return, sizeof/flags constants)
      // is emitted in C++, NOT driven by the pre/invoke calls below — those
      // document intent and gate the call-site block.  `emit ident` /
      // `emit global_tid` are functional (create the ident global + gtid).
      // Not testable yet: ClangIR does not emit omp.task, and the ABI
      // constants in outlineConstruct still need verification.
      construct task {
        outline_signature = task_entry(global_tid, task);
        capture_strategy = "packed";
        task_alloc_function = "__kmpc_omp_task_alloc";
        pre {
          emit ident;
          emit global_tid;
          emit task_alloc;   // emit alloc, bind `task` to its result, fill shareds
        }
        invoke {
          call "__kmpc_omp_task"(ident, global_tid, task);
        }
      }
    }


runtime libgomp {
  construct parallel {
    outline_signature = closure(env_ptr);
    capture_strategy = "packed";
    pre {}
    invoke {
      when has(num_threads) => call "GOMP_parallel"(body, env_ptr, num_threads, 0);
      otherwise             => call "GOMP_parallel"(body, env_ptr, 0, 0);
    }
  }
  construct wsloop when schedule == static {
    thread_id_function  = "omp_get_thread_num";
    num_thread_function = "omp_get_num_threads";
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
  construct wsloop when schedule == dynamic {
    chunk_start_function = "GOMP_loop_dynamic_start";
    chunk_next_function  = "GOMP_loop_dynamic_next";
    pre {
      emit chunked_loop;
    }
    invoke {
      emit loop_body;
    }
    post {
      when nowait => call "GOMP_loop_end_nowait"();
      otherwise   => call "GOMP_loop_end"();
    }
  }
  construct barrier {
    invoke {
      call "GOMP_barrier"();
    }
  }

  // libgomp submits a task with a single GOMP_task call, reusing the packed/
  // closure outlining path. Wired in OmpToOmpLowerPass + outlineConstruct, but
  // not yet testable end-to-end: ClangIR does not emit omp.task yet.
  construct task {
    outline_signature = closure(env_ptr);
    capture_strategy = "packed";
    invoke {
      when has(if_clause) => call "GOMP_task"(body, env_ptr, null, env_size, env_align, if_clause, 0, null, 0);
      otherwise           => call "GOMP_task"(body, env_ptr, null, env_size, env_align, true,      0, null, 0);
    }
  }
}


runtime pmsis {
  construct parallel {
    outline_signature = closure(env_ptr);
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
  // Guardless: catches static / default — inline DIVMOD distribution.
  construct wsloop {
    thread_id_function  = "ext_pi_core_id";
    num_thread_function = "ext_pi_cl_nb_cores";
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

  // Dynamic: same chunked (start/next) mechanism as libgomp, with pi_-prefixed
  // runtime functions (to be provided by the PMSIS-side implementation).
  // Reuses the generic `emit chunked_loop` path — no C++ change.
  construct wsloop when schedule == dynamic {
    chunk_start_function = "pi_GOMP_loop_dynamic_start";
    chunk_next_function  = "pi_GOMP_loop_dynamic_next";
    pre {
      emit chunked_loop;
    }
    invoke {
      emit loop_body;
    }
    post {
      when nowait => call "pi_GOMP_loop_end_nowait"();
      otherwise   => call "pi_GOMP_loop_end"();
    }
  }

  // PMSIS has no task runtime on the cluster (fork-join only), so a task runs
  // inline on the encountering core: the body is outlined and called
  // synchronously via `call body(env_ptr)`. outlineConstruct detects the
  // outlined-fn callee and emits a direct func.call. Correct for independent
  // tasks; serialises ones a real runtime would run concurrently.
  // Not testable yet: ClangIR does not emit omp.task.
  construct task {
    outline_signature = closure(env_ptr);
    capture_strategy = "packed";
    invoke {
      call body(env_ptr);
    }
  }
}
