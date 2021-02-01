# Remove end-line whitespace
s/ +$//
s/\\m(left|right)/\\\1/g
/alignat/ {
    s/\{alignat(\*?)\}/{align\1}/;
    s/\{[[:digit:]]+\}//
}
# HACK around pandoc wipe of algorithm environment: wrap in lstlisting
/\\begin\{algorithm\}/i \\n\n\\iffalse\n\\begin{lstlisting}[language=algorithm]\n\\fi
/\\end\{algorithm\}/a \\\iffalse\n\\end{lstlisting}\n\\fi\n\n
# wrap `\eqref`s in verbatims?
# /\\eqref/ s/\\eqref\{[^}]*\}/\\begin{verbatim}&\\end{verbatim}/g
/\\eqref/ s/\\eqref\{[^}]*\}/$&$/g
# Remove some macros because pandoc doesn't like them
# /\\newcommand\{\\indic\}/d
/newcommand/ {
    s/\\mathbbm/\\mathbb/g
    /\{\\numberthis\}/d
}
# TODO change align* with `\numberthis \label{}` to `\nonumber` on `\\` lines
