/``` /,/# <<begin/ {/``` /!d}                # cut code before marker
/# <<end/,/```/ {/```/!d}                    # cut code after marker
/aligned/ s/aligned/align\*/g                # rename environments
s/\\intertext\{([^}]+)\}/\\end{align*}$$ \1 $$\\begin{align*}/
/\$\$/ {
    s/(^| )\$\$/\n\nXXmath_openXX\n/g;       # place opening math delims on own line
    s/([^ ])\$\$/\1\n$$\n\n/g;               # place closing math delims on own line
    s/XXmath_openXX/$$/g
}
/\\end/ s/\\end/\n\0/g                        # place \end{} on new line
/qedhere/ s/\\\qedhere/\xE2\x97\xBB/         # replace qed with actual square
/^\xE2\x97\xBB$/d                 # remove lines with only square (inside tag instead)
# s/^::: \.algorithm/<div class="algorithm">/  # create algorithm divs
# s@^:::$@</div>@
/<div/ s/>/ markdown=1>/                     # allow markdown to work in divs
