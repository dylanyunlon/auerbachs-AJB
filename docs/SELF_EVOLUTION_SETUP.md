# Claude Self-Evolution CI/CD Setup

## Architecture

```
Trigger (dispatch / issue / cron)
  |
  v
Claude Code dev --> git push branch --> Create PR (label: auto-merge)
                                           |
                                           v
                                    auto-merge workflow
                                    +-- ci_guard.py (5 checks)
                                    +-- author allowlist
                                    +-- auto approve
                                    +-- squash merge --> main
```

## Done

- [x] ANTHROPIC_API_KEY secret
- [x] AUTO_MERGE_TOKEN secret

## Still needed

### 1. Actions write permission

Settings > Actions > General > Workflow permissions:
- Select "Read and write permissions"
- Check "Allow GitHub Actions to create and approve pull requests"

### 2. Enable auto-merge

Settings > General > Pull Requests:
- Check "Allow auto-merge"

### 3. Apply patch and push

```bash
git am 0001-ci-Claude-self-evolution-pipeline-CI-guard.patch
git push origin main
```

## Usage

Manual:   Actions > "Claude Self-Evolution" > Run workflow > fill task
Issue:    Create issue with "claude-task" label
Cron:     Uncomment schedule in workflow

## CI Guard (scripts/ci_guard.py)

1. Ban _v2/_port/_copy/_new and 30+ other suffixes
2. Ban duplicate basenames exceeding upstream count
3. New files must use ajb_ prefix or be in tests/tools
4. upstream/ is read-only
5. Same-name file diff rate >= 20%
