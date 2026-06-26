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
  construct barrier {
    invoke {
      call "GOMP_barrier"();
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
}
