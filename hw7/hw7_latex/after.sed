# Cut code, but ignore algorithm environment
/^``` algorithm$/! {
    /^``` /,/^# <<begin/ {/^``` /!d}
    /^# <<end/,/^```$/ {/^```$/!d}
}
# rename environments
/(begin|end)\{aligned\}/ s//\1{align}/g
# TODO would like to join/substitute intertext *before* pandoc, so that pandoc
# parses the text as, well... text, but we don't know what kind of environment
# surrounds it in general (align*, alignat, etc.)
s/\\intertext\{(.+)\}$/\\end{align}$$ \1 $$\\begin{align}/
# This line should only apply to text from `intertext` since it occurs after
# pandoc has parsed the rest of the file
/\\emph\{([^}]*)\}/ s//*\1*/g
# place opening/closing math delims on own line
/\$\$/ {
    s/(^| )\$\$/\n\nXXmath_openXX\n/g
    s/([^ ])\$\$/\1\n$$\n\n/g
    s/XXmath_openXX/$$/g
}
# place \end{} on new line
/([^[:blank:]])(\\end)/ s//\1\n\2/g
# Parse algorithm environment
/^``` algorithm$/,/^```$/ {
    # Convert caption and label into header
    /^``` algorithm$/ {
        :x
        N
        /\\label/ { 
            # perform substitions
            s@(.*)\\caption\{(.*)\}[[:space:]]*\\label\{(.*)\}@\
<div class="algorithm" id="\3">\
<div class="alg_caption_div">\
<span class="alg_title">Algorithm</span>\
<span class="alg_caption">\2</span>\
</div>@
            # then break out
            b
        }
        b x
    }
    # Remove tex lines
    /\\iffalse/d
    /\\fi/d
    # remove comments 
    s/([^\\])%.*$/\1/
    # Format algorithmic commands
    /\\begin\{algorithmic\}/,/\\end\{algorithmic\}/ {
        # un-escape commands
        s/\\([A-Z][[:alpha:]]+)/\1/g
        s@(Require|Ensure)@<span class="alg_command">\0:</span>@
        s@(Procedure|Function)\{([^}]+)\}\{([^}]+)\}@<span class="alg_command">\u\1</span> <span class="alg_proc">\2</span>(\3)@
        s@Call\{([[:alnum:]]+)\}\{(\$.+\$)\}@<span class="alg_call">\1</span>(\2)@
        s@ForAll\{(.+)\}$@<span class="alg_command">for all</span> \1 <span class="alg_command">do</span>@
        s@End(For|Procedure|Function)@<span class="alg_command">end \L\1</span>@
        s/State *//
        s@(Return|Assert)@<span class="alg_command">\u\1</span>@
        # TODO define comment style and character in HTML class
        s@Comment\{(.*)\}$@<span style="float: right">\&#x25B7; \1</span>@g
        # Change spaces to tabs for nice indenting
        s/^ {4}//
        s/ {2}/\t/g
        # wrap every non-blank line in a paragraph
        /[^[:blank:]]/ s@.*@<p>&</p>@
        # sub normal macros just this once...
        s/\\N\{([^}]*)\}\{([^}]*)\}/\\mathcal{N}\\left( \1, \2 \\right)/g
    }
    # Make algorithmic class for formatting the paragraph
    s/^.*\\begin\{algorithmic\}.*$/<div class="algorithmic">/
    s@^.*\\end\{algorithmic\}.*$@</div>@
    # Drop the latex statements
    /\\(begin|end)/d
    # Replace final div marker
    s|^```$|</div>|
}
# allow markdown to work in divs
/<div/ s/>/ markdown=1>/
# remove spaces before footnotes
/\[\^[0-9]+\]/ s/\. (\[\^[0-9]+\])/.\1/g
# Update figure references to make link encompass `Figure X`
/(Figure|Proposition|Theorem|Lemma|Table)[[:blank:]]<a/ {
    s//<a/g;
    s@>([0-9]+)<@>Figure \1<@g
}
# fix ceil/floor\*
/\\(ceil|floor)\*?/ s//\\\1/g

