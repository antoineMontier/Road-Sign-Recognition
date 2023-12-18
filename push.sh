#!/bin/bash
# pre-activation command : chmod 777 push.sh

function git_add_commit_push() {
    if (($# == 0)) ; then
        echo "error, use this way: './push.sh my git commit message'"
    else
        git add *.py *.ipynb *.md

        commit_message="git commit -m '"
        for word in "$@" ; do
            commit_message="$commit_message $word"
        done
        commit_message="$commit_message'"

        eval "$commit_message"
        git push 
    fi
}

git_add_commit_push "$@"
