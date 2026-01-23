---
description: Complete deployment pipeline that persists until done
---

# Deploy Command

## Usage

```
/deploy [environment]
```

Optional `environment` argument specifies target (e.g., `staging`, `production`, `dev`). If omitted, detect from environment variables or project configuration.

## Inputs / Assumptions

- Detect project type via manifests: `@package.json`, `@pyproject.toml`, `@Cargo.toml`, `@go.mod`, etc.
- Detect deployment target/environment from:
  - Environment variables (`DEPLOY_ENV`, `NODE_ENV`, `ENVIRONMENT`)
  - Configuration files (`.env`, deployment configs)
  - Git branch name (if applicable)
- Assume deployment tools are configured (e.g., Vercel, AWS, Docker, etc.)

## Steps

1. **Run fast checks first**
   - Execute `!make check` or equivalent fast-check script if available
   - If checks fail, report failures and ask user if they want to proceed anyway
   - Document in post-mortem if checks are skipped due to deployment urgency

2. **Detect environment**
   - Check environment variables: `!env | grep -E "(ENV|ENVIRONMENT|DEPLOY)"`
   - Read deployment config files if present
   - If `$ARGUMENTS` provided, use that as environment name
   - Report detected environment to user

3. **Build project**
   - Detect build command from package manifest or Makefile
   - Execute build: `!npm run build`, `!make build`, `!cargo build --release`, etc.
   - If build fails:
     - Gather build logs
     - Analyze error messages
     - Propose fix
     - Retry build
     - If 3+ retries needed, document in post-mortem (category: DEPLOY)

4. **Deploy**
   - Execute deployment command based on detected platform
   - Monitor deployment output for errors
   - If deployment fails:
     - Gather deployment logs
     - Identify failure cause (timeout, size limits, permissions, etc.)
     - Propose fix
     - Request user confirmation before retry if 2+ failures
     - Retry deployment
     - If 3+ retries or >15min resolution time, document in post-mortem (category: DEPLOY)

5. **Verify deployment**
   - Run health checks:
     - Check deployment endpoint: `!curl -f <deployment-url>/health` or equivalent
     - Verify environment variables are set correctly
     - Check deployment logs for startup errors
   - If health check fails:
     - Gather diagnostic information
     - Offer rollback option to user
     - If rollback needed or verification takes >15min, document in post-mortem (category: DEPLOY)

6. **Report success**
   - Display deployment URL/endpoint
   - Show deployment time
   - Confirm environment

## Outputs

- Deployment status (success/failure)
- Deployment URL/endpoint
- Build logs (if failures occurred)
- Health check results
- Rollback instructions (if verification fails)

## Struggle Tracking (When to Document)

Document in `@docs/POSTMORTEMS.md` when encountering:

- **Build failures**: Requiring 3+ retry attempts or >15min to resolve
- **Size limit issues**: Deployment rejected due to bundle/artifact size exceeding limits
- **Health check failures**: Verification failures requiring multiple attempts or rollback
- **Environment mismatches**: Configuration mismatches between local and deployment environments requiring 3+ retries or >15min to resolve
- **Tool failures**: Deployment tool errors requiring workarounds

**Criteria**: 3+ retries OR >15min resolution time OR workaround required

**Category**: DEPLOY

## Post-Mortem Write Path

Before documenting a struggle:

1. Check `@docs/POSTMORTEMS.md` for similar entries using:
   - `!rg -n -i "deploy\|build\|health" docs/POSTMORTEMS.md`
   - Search by category: `!rg -n "DEPLOY" docs/POSTMORTEMS.md`

2. If duplicate found:
   - Ask user: "Similar entry found: [entry]. Update existing or add new?"
   - If update: Modify existing entry with additional context
   - If new: Add new entry (newest first)

3. If new entry:
   - Append to `@docs/POSTMORTEMS.md` in the "New Entries" section
   - Use standard format: date, category (DEPLOY), title, task, struggle, resolution, time lost, prevention
   - Commit with message: `Document DEPLOY struggle in POSTMORTEMS.md`
