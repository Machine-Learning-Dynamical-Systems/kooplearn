set shell := ["bash", "-c"]

# Usage: just release [major|minor|patch]
release part="": check-clean check-branch
    #!/usr/bin/env bash
    set -e

    # 1. Bump Version
    if [ -n "{{part}}" ]; then
        echo "🚀 Bumping version ({{part}})..."
        uv version --bump "{{part}}"
    else
        echo "ℹ️  No bump argument provided. Using current."
    fi

    # 2. Capture Version (Dynamic)
    NEW_VERSION=$(uv version --short | awk '{print $NF}')
    echo "📦 Releasing version: $NEW_VERSION"

    # 3. Git Operations
    git add pyproject.toml uv.lock 2>/dev/null
    git commit -m "Release v$NEW_VERSION" || echo "Nothing to commit, tagging existing..."
    
    git tag "v$NEW_VERSION"

    echo "⬆️  Pushing to GitHub..."
    git push origin main
    git push origin "v$NEW_VERSION"
    
    echo "✅ Done! v$NEW_VERSION pushed."

# --- Safety Checks ---

[private]
check-clean:
    @if [ -n "$(git status --porcelain)" ]; then echo "❌ Error: Dirty workspace."; exit 1; fi

[private]
check-branch:
    @if [ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]; then echo "❌ Error: Not on main branch."; exit 1; fi
    
# Run all tests
test-all:
    uv run pytest -s tests

# Build docs:
docs:
    uv run sphinx-autobuild -a -E docs docs/_build/html

