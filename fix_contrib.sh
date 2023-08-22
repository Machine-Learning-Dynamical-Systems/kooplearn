#!/bin/bash
git filter-branch -f --env-filter '
WRONG_EMAIL=“gregoire.pacreau@hotmail.fr”
NEW_NAME=“GregoirePacreau” # github username
NEW_EMAIL=“gregoirepacreau@hotmail.fr”
if [ “$GIT_COMMITTER_EMAIL” = “$WRONG_EMAIL” ]
then
    export GIT_COMMITTER_NAME=“$NEW_NAME”
    export GIT_COMMITTER_EMAIL=“$NEW_EMAIL”
fi
if [ “$GIT_AUTHOR_EMAIL” = “$WRONG_EMAIL” ]
then
    export GIT_AUTHOR_NAME=“$NEW_NAME”
    export GIT_AUTHOR_EMAIL=“$NEW_EMAIL”
fi
' --tag-name-filter cat -- --branches --tags