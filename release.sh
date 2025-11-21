#!/bin/bash

# 1. Check for arguments
if [ -z "$1" ]; then
  echo "Error: Please specify a bump type (patch, minor, major, etc.)"
  echo "Usage: ./release.sh patch"
  exit 1
fi

# 2. Safety Check: Ensure working directory is clean
# You don't want to accidentally tag uncommitted junk files.
if [ -n "$(git status --porcelain)" ]; then
  echo "Error: Working directory is not clean. Commit or stash changes first."
  exit 1
fi

# 3. Safety Check: Ensure we are on main
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != "main" ]]; then
  echo "Error: You are on branch '$BRANCH'. Please switch to main."
  exit 1
fi

echo "🚀 Bumping version..."

# 4. Bump the version using uv
uv version --bump "$1"

# 5. Capture the new version
# We use awk to ensure we only get the version number (e.g. "0.1.1") 
# removing project name if uv prints "kooplearn 0.1.1"
NEW_VERSION=$(uv version  --short --dry-run | awk '{print $NF}')

echo "📦 New version is: $NEW_VERSION"

# 6. Commit, Tag, and Push
git add pyproject.toml uv.lock # Add lockfile if you use one
git commit -m "Release v$NEW_VERSION"
git tag "v$NEW_VERSION"

echo "⬆️  Pushing to GitHub..."
git push
git push --tags

echo "✅ Done! GitHub Actions will now build and publish v$NEW_VERSION."