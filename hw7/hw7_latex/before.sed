s/\\m(left|right)/\\\1/g
/alignat/ {
    s/\{alignat(\*?)\}/{align\1}/;
    s/\{[[:digit:]]+\}//
}
# wrap in lstlisting
/\\begin\{algorithm\}/i \\n\n\\iffalse\n\\begin{lstlisting}[language=algorithm]\n\\fi
/\\end\{algorithm\}/a \\\iffalse\n\\end{lstlisting}\n\\fi\n\n
# Remove some macros because pandoc doesn't like them
# /\\newcommand\{\\indic\}/d
/newcommand/ {
    s/\\mathbbm/\\mathbb/g
    /\{\\numberthis\}/d
}
