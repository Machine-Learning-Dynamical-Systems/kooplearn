#!/bin/bash

# 1. Safety Check: Clean working directory
if [ -n "$(git status --porcelain)" ]; then
  echo "Error: Working directory is not clean. Commit or stash changes first."
  exit 1
fi

# 2. Safety Check: Branch is main
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != "main" ]]; then
  echo "Error: You are on branch '$BRANCH'. Please switch to main."
  exit 1
fi

# 3. Conditionally Bump Version
if [ -n "$1" ]; then
  echo "🚀 Bumping version ($1)..."
  uv version --bump "$1"
else
  echo "ℹ️  No bump argument provided. Using current version."
fi

# 4. Capture Version
NEW_VERSION=$(uv version --short | awk '{print $NF}')
echo "📦 Releasing version: $NEW_VERSION"

# 5. Commit (if changed), Tag, and Push
git add pyproject.toml uv.lock 2>/dev/null
# Allow commit to fail if there are no changes (e.g., re-tagging same version)
git commit -m "Release v$NEW_VERSION" || echo "Nothing to commit, proceeding to tag..."

git tag "v$NEW_VERSION"

echo "⬆️  Pushing to GitHub..."
git push
git push --tags

echo "✅ Done! v$NEW_VERSION pushed."