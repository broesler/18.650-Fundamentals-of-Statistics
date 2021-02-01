# Remove end-line whitespace
s/ +$//
# Remove unrecognized macros
s/\\m(left|right)/\\\1/g
/alignat/ {
    s/\{alignat(\*?)\}/{align\1}/;
    s/\{[[:digit:]]+\}//
}
# HACK around pandoc wipe of algorithm environment: wrap in lstlisting
/\\begin\{algorithm\}/i \\n\n\\iffalse\n\\begin{lstlisting}[language=algorithm]\n\\fi
/\\end\{algorithm\}/a \\\iffalse\n\\end{lstlisting}\n\\fi\n\n
# wrap `\equations`s in verbatim
/\\begin\{equation\}/i \\n$$\n
/\\end\{equation\}/a \\n$$\n
/\\eqref/ s/\\eqref\{[^}]*\}/$&$/g
# Remove some macros because pandoc doesn't like them
# /\\newcommand\{\\indic\}/d
/newcommand/ {
    s/\\mathbbm/\\mathbb/g
    /\{\\numberthis\}/d
}
