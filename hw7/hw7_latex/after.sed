# Cut code, but ignore algorithm environment
/^``` algorithm$/! {
    /^``` /,/^# <<begin/ {/^``` /!d}
    /^# <<end/,/^```$/ {/^```$/!d}
}
# rename environments
/aligned/ s/(begin|end)\{aligned\}/\1{align*}/g
s/\\intertext\{([^}]+)\}/\\end{align*}$$ \1 $$\\begin{align*}/
# place opening/closing math delims on own line
/\$\$/ {
    s/(^| )\$\$/\n\nXXmath_openXX\n/g
    s/([^ ])\$\$/\1\n$$\n\n/g
    s/XXmath_openXX/$$/g
}
# place \end{} on new line
/\\end/ s/([^[[:blank:]])(\\end)/\1\n\2/g
# Parse algorithm environment
/^``` algorithm$/,/^```$/ {
    # remove comments and extraneous lines
    s/([^\\])%.*$//
    /\\iffalse/d
    /\\fi/d
    # create algorithm divs
    s/^``` algorithm/<div class="algorithm">/
    s|^```$|</div>|
    # TODO: caption, label
    /\\begin\{algorithmic\}/,/\\end\{algorithmic\}/ {
        # un-escape commands
        s/\\([A-Z][[:alpha:]]+)/\1/g
        s@(Require|Ensure)@<strong>\0:</strong>@
        s@(Procedure|Function)\{([^}]+)\}\{([^}]+)\}@<strong>\u\1</strong> <span style="font-variant: small-caps">\2</span>(\3)@
        s@Call\{([[:alnum:]]+)\}\{(\$.+\$)\}@<span style="font-variant: small-caps">\1</span>(\2)@
        s@ForAll\{(.+)\}$@<strong>for all</strong> \1 <strong>do</strong>@
        s@End(For|Procedure|Function)@<strong>end \L\1</strong>@
        s/State//
        s@(Return|Assert)@<strong>\u\1</strong>@
        s@Comment\{([^}]+)\}@<span style="float: right">\&#x25B7; \1</span>@g
        # wrap everything in a paragraph
        s@.*@<p class="algorithmic"> & </p>@
    }
    /\\(begin|end)/d
}
# allow markdown to work in divs
/<div/ s/>/ markdown=1>/
