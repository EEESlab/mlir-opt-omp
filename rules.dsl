runtime iomp {
  global_tid_function = "__kmpc_global_thread_num";

  // `chunk` argument of both work-sharing entry points
  let default_chunk = 1;

  construct parallel {
    // ABI selector: microtask signature void(ptr gtid, ptr btid, cap0, ...)
    capture_strategy = by_pointer;

    pre {
      when has(num_threads) => call "__kmpc_push_num_threads"(ident, global_tid, num_threads);
      when has(proc_bind) => call "__kmpc_push_proc_bind"(ident, global_tid, proc_bind);
    }

    invoke {
      // `branch` (not `when`) because if(cond) is a runtime value
      branch if_clause {
        true  => call "__kmpc_fork_call"(ident, argc(captures), body, captures);
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

  construct wsloop when schedule == dynamic {
    pre {
      when has(chunk) => call "__kmpc_dispatch_init_4"(ident(work_loop), global_tid, 35, lower_val, upper_incl, step, chunk);
      otherwise       => call "__kmpc_dispatch_init_4"(ident(work_loop), global_tid, 35, lower_val, upper_incl, step, default_chunk);
    }
    next_chunk {
      call "__kmpc_dispatch_next_4"(ident(work_loop), global_tid, last, lower, upper, stride);
    }
    invoke {
      emit loop_body;
    }
    post {
      when not nowait => call "__kmpc_barrier"(ident(barrier_impl_for), global_tid);
    }
  }

  construct task {
    // ABI selector: task routine i32(i32 gtid, ptr task), captures in a
    // runtime-allocated shareds struct reached via load(task->shareds).
    capture_strategy = shareds;
    kmp_task_t = struct(ptr, ptr, i32, ptr, ptr);
    let task_flags = 1; 

    invoke {
      let task = call "__kmpc_omp_task_alloc"(ident, global_tid, task_flags,
                                              task_size, shareds_size, body);
      emit populate_shareds(task);

      branch if_clause {
        true  => call "__kmpc_omp_task"(ident, global_tid, task);
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
    // ABI selector: closure void(ptr data), all captures in one struct.
    capture_strategy = packed;
    pre {}
    invoke {
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
  construct wsloop when schedule == dynamic {
    chunk_index  = i64;
    chunk_result = i8;
    chunk_bound  = exclusive;

    let default_chunk = 1;

    first_chunk {
      when has(chunk) => call "GOMP_loop_dynamic_start"(lower_val, upper_val, step, chunk, lower, upper);
      otherwise       => call "GOMP_loop_dynamic_start"(lower_val, upper_val, step, default_chunk, lower, upper);
    }
    next_chunk {
      call "GOMP_loop_dynamic_next"(lower, upper);
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
  construct taskwait {
    invoke {
      call "GOMP_taskwait"();
    }
  }
  construct task {
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
    capture_strategy = packed;
    let default_team_size = 8; 
    pre {}
    invoke {
      // ext_pi_cl_team_fork has no `if` parameter, so the clause lowers to a
      // one-core team. Forking a team of 1 rather than calling the closure
      // directly keeps an ext_pi_cl_team_barrier in the body well-formed.
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
  // The guard matters even as the only variant: the cluster has no dispatch
  // API, so without it a dynamic schedule would silently lower as static
  // instead of being rejected.
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
