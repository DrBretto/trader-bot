# Trusted User Sharing Plan

Date: 2026-03-13

## Context

- This system is not intended to become a public product.
- Planned audience is limited to the operator, a few close friends, and one known outside evaluator.
- Users will be told the system state plainly, including that broker automation is still in a controlled rollout stage.
- Short-term execution path remains unchanged: paper soak first, then operator live cutover later.

## Objective

Define the cleanest future path for allowing a small set of trusted people to connect their own Alpaca accounts without turning the app into a general-purpose brokerage product.

## Decisions

1. End-user sign-in should use Amazon Cognito Hosted UI with Google sign-in.
   - Do not use AWS account login as the user-facing auth path.
   - Reason: AWS login is an operator/admin tool, not a good end-user experience, and session churn is unnecessary friction.

2. Broker linking should use Alpaca OAuth / Connect per user.
   - Do not ask users to paste API keys into the app.
   - Reason: this keeps brokerage authorization separate from app identity and fits the intended multi-user shape.

3. Keep app identity and broker identity separate.
   - App identity: Cognito user (`sub`)
   - Broker identity: linked Alpaca paper/live account for that user

4. Keep paper and live as explicit account contexts.
   - The UI should expose a real context toggle, not a fake display-only switch.
   - Expected contexts:
     - `paper`
     - `live`

5. Preserve the option to keep paper public while protecting live later.
   - Public paper view is fine for demo/showcase use.
   - Live views and live broker actions should eventually sit behind Cognito auth.

## Recommended Architecture

### Identity

- Amazon Cognito Hosted UI
- Google sign-in federation
- Standard session refresh flow handled by Cognito

### Broker authorization

- Alpaca OAuth / Connect flow
- One broker link per user account, with separate paper/live mode metadata

### Storage

- DynamoDB:
  - user registry
  - linked account registry
  - per-user mode flags
  - per-user risk/kill-switch metadata
- AWS Secrets Manager:
  - Alpaca refresh tokens
  - any other broker-sensitive credentials
- AWS KMS:
  - encryption backing for sensitive secrets and data-at-rest controls

### UI / routing model

- Public demo path can continue to show paper results.
- Authenticated users should get their own account-scoped views.
- Later operator tooling can support viewing multiple linked accounts if needed.

## Proposed Phases

### Phase A: Current state

- Single operator account
- Alpaca paper mode
- Operational soak and drift monitoring

### Phase B: Operator live cutover

- Add live credentials
- Add live/paper account-context switch
- Keep paper available for public/demo viewing

### Phase C: Trusted-user onboarding

- Add Google sign-in via Cognito
- Add Alpaca connect/link flow
- Add per-user account registry and mode selection
- Add user-level kill switch and risk caps

### Phase D: Privacy hardening

- Protect live views behind Cognito auth
- Keep public paper view only if still useful

## Required Capabilities Before Trusted-User Sharing

- Per-user registry keyed by Cognito `sub`
- Secure broker token storage
- User-scoped broker execution context
- User-scoped portfolio/dashboard data resolution
- User-level kill switch
- Audit trail for broker connect/disconnect and live-mode changes

## Constraints and Risks

- Alpaca may require additional approval for external-user live trading through OAuth-based app flows.
- Multi-user support introduces data partitioning risk; every broker action and dashboard artifact must remain user-scoped.
- Public sharing must never expose another user's live account by accident.

## Recommendation

Build the future sharing path around `Cognito + Google sign-in` for user identity and `Alpaca OAuth / Connect` for broker authorization.

Do not build around AWS login for end users.

Do not schedule implementation until paper soak is complete and the operator live cutover gate is defined.
