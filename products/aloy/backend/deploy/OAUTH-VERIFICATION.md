# Google OAuth verification — the path to public users

_The longest-lead item on the ship track. Until verification completes, the
Gmail/Calendar connect flow works only for up to 100 test users, each seeing
an "unverified app" warning. Start this early; the clock is mostly Google's._

## Where we stand

- OAuth client exists (`GOOGLE_OAUTH_CLIENT_ID/SECRET`), consent screen in
  **Testing** mode — test users you add by email work today.
- Scopes the connect-engine requests (see `connections/providers.py`):
  Gmail read/send + Calendar. **Gmail scopes are RESTRICTED scopes** — the
  heaviest verification tier, including a third-party security assessment
  (CASA) once you pass 100 users. Calendar scopes are "sensitive" (verification
  but no CASA).

## Tiers — decide how far to go, in order

1. **Stay in Testing (now, free)** — up to 100 named test users, warning
   screen, 7-day refresh-token expiry on some account types. Fine for
   friends-and-family. Action: none.
2. **Publish + verify sensitive scopes (weeks)** — removes the warning for
   Calendar-only. Gmail still restricted.
3. **Restricted-scope verification incl. CASA (months, can cost $)** — full
   public Gmail access. Only worth it with real traction. CASA Tier 2
   self-scan options exist (e.g. Fortify/OWASP ZAP based) that keep cost low.

**Recommendation: do (1) now, prepare (2)'s assets (they're needed anyway),
defer (3) until users demand Gmail at scale.** An alternative worth pricing
later: route ONLY Gmail through a verified aggregator while keeping Calendar
native.

## The asset checklist (needed for any tier above Testing)

All of these must live on the **authorized domain** (the domain of
`APP_BASE_URL` / the consent screen):

- [ ] **Homepage** that explains what Aloy does — exists
      (`products/aloy/website`).
- [ ] **Privacy policy URL** — `products/aloy/website/privacy.html`
      (added by this PR; deploy with the site).
- [ ] **Terms of service URL** — `products/aloy/website/terms.html` (same).
- [ ] The privacy policy must specifically describe **what Google user data
      is accessed and how it's used/stored/shared** — the drafted policy has
      a dedicated "Google user data" section; keep it accurate when scopes
      change.
- [ ] **Limited Use disclosure** on the site (Google API Services User Data
      Policy compliance statement) — included in privacy.html.
- [ ] Consent screen: app name, logo (120x120), support email, authorized
      domain(s), links to the two pages above.
- [ ] **Domain verification** in Google Search Console for the authorized
      domain.
- [ ] **Demo video** (YouTube, unlisted OK): show the full OAuth flow from
      the consent screen through the feature that uses each requested scope
      (search inbox, send a mail, list calendar). Narrate scope→feature
      mapping.
- [ ] **Scope justification text**: one paragraph per scope explaining why a
      narrower scope is insufficient. Keep with the submission.

## Console steps (operator — needs your Google account)

1. Google Cloud Console → APIs & Services → OAuth consent screen.
2. Fill branding (name, logo, support email), add authorized domain, link
   privacy + terms URLs.
3. Verify the domain in Search Console (DNS TXT record).
4. Add test users (Testing tier) OR hit **Publish app** → **Prepare for
   verification** and submit the checklist above.
5. Respond to Google's review emails promptly — each round-trip costs days.

## Redirect URIs (do alongside deployment)

The prod backend handles the OAuth redirect. In the OAuth client config add:

- `https://api.pori.aloysathekge.com/v1/connections/google/callback`
  (verified: `routes/connections.py` builds
  `{BACKEND_BASE_URL}/v1/connections/{provider}/callback`)

and keep the localhost one for dev. `BACKEND_BASE_URL` in SSM must match the
prod origin exactly.
