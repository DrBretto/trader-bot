---
description: Performance profiling and optimization
---

# Performance Command

## Usage

```
/performance [target]
```

Optional `target` argument specifies what to profile:
- `bundle` - Analyze bundle sizes (web projects)
- `runtime` - Profile runtime performance (backend/API)
- `build` - Analyze build performance
- If omitted, detect project type and profile accordingly

## Inputs / Assumptions

- Detect project type:
  - Web frontend: `@package.json` with bundler (webpack, vite, rollup)
  - Backend/API: Server code, API routes
  - Full-stack: Both frontend and backend
- Detect available profiling tools:
  - Web: `webpack-bundle-analyzer`, `source-map-explorer`, Lighthouse
  - Node.js: `clinic.js`, `0x`, Node.js built-in profiler
  - Python: `cProfile`, `py-spy`, `memory_profiler`
  - Rust: `cargo flamegraph`, `perf`
  - Go: `go tool pprof`, `go tool trace`

## Steps

1. **Detect project type and profiling tools**
   - Check `@package.json` for bundler and profiling tools
   - Check for profiling config files
   - Report detected project type and available tools

2. **For web projects: Analyze bundle sizes**
   - Build project: `!npm run build` or equivalent
   - Analyze bundle:
     - If `webpack-bundle-analyzer` available: `!npx webpack-bundle-analyzer build/stats.json`
     - If `source-map-explorer` available: `!npx source-map-explorer 'build/static/js/*.js'`
     - Otherwise: Analyze `build/` directory sizes: `!du -sh build/**/*`
   - Identify large dependencies
   - Check code splitting:
     - Verify route-based code splitting
     - Check for duplicate dependencies across chunks
   - Generate bundle report:
     - Largest files/packages
     - Total bundle size
     - Code splitting effectiveness

3. **For backend/API: Profile runtime performance**
   - Identify slow functions:
     - If Node.js: Run with profiler: `!node --prof app.js` then analyze
     - If Python: Use `cProfile`: `!python -m cProfile -o profile.stats app.py`
     - If Rust: Use `cargo flamegraph`: `!cargo flamegraph`
     - If Go: Use `go tool pprof`: `!go tool pprof http://localhost:6060/debug/pprof/profile`
   - Identify slow database queries:
     - Check query logs
     - Look for N+1 queries
     - Check for missing indexes
   - Generate performance profile:
     - Slowest functions
     - CPU usage hotspots
     - Memory usage patterns
     - Database query performance

4. **Suggest optimizations**
   - **For web projects**:
     - Lazy loading: Identify routes/components that can be lazy loaded
     - Code splitting: Suggest additional split points
     - Tree shaking: Identify unused code that can be removed
     - Dependency optimization: Replace large dependencies with lighter alternatives
     - Image optimization: Compress images, use modern formats
   - **For backend**:
     - Caching: Identify cacheable data/queries
     - Database optimization: Suggest indexes, query optimizations
     - Algorithm improvements: Identify inefficient algorithms
     - Resource pooling: Connection pooling, worker pools
   - Prioritize by:
     - Impact (high/low)
     - Effort (low/medium/high)
     - Prefer high-impact, low-effort optimizations

5. **Generate performance report**
   - Current metrics:
     - Bundle sizes (web) or runtime metrics (backend)
     - Performance bottlenecks identified
   - Optimization recommendations:
     - Prioritized list of optimizations
     - Estimated impact
     - Implementation effort
   - Before/after comparison (if metrics available from previous runs)

6. **Apply optimizations** (if user confirms)
   - Apply selected optimizations
   - Re-measure performance
   - Compare before/after metrics
   - Document improvements

## Outputs

- Performance report with:
  - Current performance metrics
  - Identified bottlenecks
  - Prioritized optimization recommendations
  - Before/after comparison (if available)
  - Implementation guide for selected optimizations

## Struggle Tracking (When to Document)

Document in `@docs/POSTMORTEMS.md` when encountering:

- **Profiling tools unavailable**: Required profiling tools not installed or unavailable requiring manual setup
- **Unclear performance bottleneck**: Performance issue identified but root cause unclear requiring investigation
- **Optimization tradeoffs**: Optimizations have unclear tradeoffs requiring analysis (e.g., bundle size vs runtime performance)
- **Profiling tool failures**: Profiling tools fail to run or produce incorrect results

**Criteria**: 3+ retries OR >15min resolution time OR workaround required

**Category**: TOOLING (or UNDERSTANDING if unclear bottleneck)

## Post-Mortem Write Path

Before documenting a struggle:

1. Check `@docs/POSTMORTEMS.md` for similar entries using:
   - `!rg -n -i "performance\|profiling\|optimization" docs/POSTMORTEMS.md`
   - Search by category: `!rg -n "TOOLING\|UNDERSTANDING" docs/POSTMORTEMS.md`

2. If duplicate found:
   - Ask user: "Similar entry found: [entry]. Update existing or add new?"
   - If update: Modify existing entry with additional context
   - If new: Add new entry (newest first)

3. If new entry:
   - Append to `@docs/POSTMORTEMS.md` in the "New Entries" section
   - Use standard format: date, category (TOOLING or UNDERSTANDING), title, task, struggle, resolution, time lost, prevention
   - Commit with message: `Document [category] struggle in POSTMORTEMS.md`
