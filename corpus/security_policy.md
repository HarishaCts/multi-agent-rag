# Northwind Security & Access Control Policy

**Source:** Security Handbook
**Author:** Security Team
**Date:** 2026-02-03
**Access:** confidential

## Authentication

All Enterprise accounts must use Single Sign-On (SSO) via SAML 2.0 or OpenID Connect.
Multi-factor authentication (MFA) is mandatory for every administrator account and is
enforced after the first login. Session tokens expire after 8 hours of inactivity.

## Data Encryption

Data is encrypted in transit using TLS 1.3. Data at rest is encrypted with AES-256.
Customer-managed encryption keys (CMEK) are available on the Enterprise edition only.

## Access Tiers

Northwind uses role-based access control (RBAC) with four built-in roles:

- **Viewer** — can view dashboards shared with them.
- **Analyst** — can create and edit dashboards and connect data sources.
- **Admin** — can manage users, billing, and workspace settings.
- **Owner** — full control including workspace deletion.

## Audit Logging

Audit logs capture every administrative action and are retained for 365 days on the
Enterprise edition. Logs can be streamed to an external SIEM via the Northwind
audit webhook. Audit logging is not available on the Starter or Team editions.

## Incident Response

Security incidents must be reported to security@northwind.example within 1 hour of
detection. The target time to acknowledge a critical incident is 30 minutes.
