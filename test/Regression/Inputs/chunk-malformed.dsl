// A rule file used by chunk-rules-rejected.mlir only, not the one the tool
// ships.  It holds four wsloop constructs, each with one thing wrong with its
// chunk blocks or properties — the mistakes a rules.dsl author can make that
// would otherwise produce silently wrong code rather than an error.
//
// They are told apart by their guard rather than by living in four files: the
// schedule kind on the input picks one, so a single run reaches all four.  The
// runtime has to be one the omp_lower.construct verifier knows, hence libgomp.

runtime libgomp {
  construct parallel {
    capture_strategy = packed;
    invoke { call "GOMP_parallel"(body, env_ptr, 0, proc_bind_flags); }
  }

  // A first_chunk with no next_chunk: the loop could open but never ask for a
  // second chunk.  Left alone this is the dangerous one — the construct does
  // not count as chunked, so the block is dropped and every thread runs the
  // whole iteration space with nothing registered with the runtime.
  construct wsloop when schedule == dynamic {
    first_chunk { call "start"(lower_val, upper_val, step, 1, lower, upper); }
    invoke { emit loop_body; }
  }

  // A next_chunk holding no call at all: nothing for the loop to turn on.
  construct wsloop when schedule == guided {
    next_chunk { emit nothing; }
    invoke { emit loop_body; }
  }

  // A misspelled chunk_bound.  Silently falling back to `inclusive` would run
  // one iteration past the end of every chunk, which is why an unreadable value
  // is an error and not a default.
  construct wsloop when schedule == auto {
    chunk_bound = exclusve;
    next_chunk { call "next"(lower, upper); }
    invoke { emit loop_body; }
  }

  // A width that is not one of the ABI types the lowering knows.
  construct wsloop when schedule == runtime {
    chunk_index = i62;
    next_chunk { call "next"(lower, upper); }
    invoke { emit loop_body; }
  }
}
