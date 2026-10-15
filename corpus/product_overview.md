# Northwind Analytics Platform — Product Overview

**Source:** Product Documentation
**Author:** Product Team
**Date:** 2026-01-15
**Access:** public

## What is Northwind Analytics?

Northwind Analytics is a cloud-native business intelligence platform that lets teams
connect data sources, build dashboards, and share insights. It is delivered as a fully
managed SaaS product and is also available as a self-hosted enterprise deployment.

## Editions

Northwind ships in three editions:

- **Starter** — up to 5 users, 10 dashboards, community support. Costs $0 per month.
- **Team** — up to 50 users, unlimited dashboards, email support. Costs $49 per user per month.
- **Enterprise** — unlimited users, SSO, audit logs, and a 99.95% uptime SLA. Pricing is custom.

## Supported Data Sources

The platform natively connects to PostgreSQL, MySQL, Snowflake, BigQuery, and Amazon S3.
A generic REST connector is available for any JSON HTTP API. The maximum number of
simultaneous data source connections on the Team edition is 25.

## Data Refresh

Scheduled refreshes run as often as every 15 minutes on the Team edition and every
5 minutes on the Enterprise edition. Starter edition refreshes once per day.
