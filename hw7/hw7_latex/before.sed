s/\\m(left|right)/\\\1/g
/alignat/ {
    s/\{alignat(\*?)\}/{align\1}/;
    s/\{[[:digit:]]+\}//
}
/\\begin\{algorithm\}/i \\n\n\\iffalse\n::: \{.algorithm\}\n\\fi
/\\end\{algorithm\}/a \\\iffalse\n:::\n\\fi\n\n
