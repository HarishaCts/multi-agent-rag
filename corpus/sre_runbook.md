# Northwind SRE Runbook

**Source:** Engineering Runbook
**Author:** SRE Team
**Date:** 2026-03-10
**Access:** internal

## Service Level Objectives

The platform targets a 99.95% monthly uptime SLO for the Enterprise edition. The error
budget for a 30-day window is therefore approximately 21 minutes of downtime.

## On-Call Rotation

On-call engineers rotate weekly. The primary on-call must acknowledge a page within
5 minutes. If unacknowledged, the alert escalates to the secondary on-call after
10 minutes, and to the engineering manager after 20 minutes.

## Common Incidents

### Dashboard load latency high
If p95 dashboard load latency exceeds 3 seconds, check the query cache hit rate first.
A cache hit rate below 70% usually indicates a cache node needs to be restarted.
Restart the cache fleet with `kubectl rollout restart deploy/query-cache`.

### Data refresh failures
If scheduled refreshes fail, inspect the connector logs. The most common cause is
expired data source credentials. Rotating the credentials in Settings > Data Sources
resolves it in most cases.

## Backups

The primary database is backed up every 6 hours. Backups are retained for 30 days.
A full restore drill is performed quarterly, and the last drill restored 2 TB in
42 minutes.

## Deployment

Production deploys happen via a blue-green strategy. A deploy is automatically rolled
back if the error rate exceeds 2% within the first 10 minutes after cutover.
