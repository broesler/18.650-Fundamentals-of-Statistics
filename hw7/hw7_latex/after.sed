/``` /,/\# <<begin/ {/``` /!d}               # cut code before marker
/\# <<end/,/```/ {/```/!d}                   # cut code after marker
/aligned/ s/aligned/align\*/g                # rename environments
/\$\$/ s/(^| )\$\$/\n\n$$\n/g                # place opening math delims on own line
s/([^ ])\$\$([[:blank:]]*|$)/\1\n$$\n\n/g    # place closing math delims on own line
/qedhere/ s/\\\qedhere/\xE2\x97\xBB/         # replace qed with actual square
/^\xE2\x97\xBB$/d                            # remove lines with only square (inside tag instead)
s/^::: \.algorithm/<div class="algorithm">/  # create algorithm divs
s@^:::$@</div>@
/mathbbm/ s/\\mathbbm\{1\}/\\indicator/g     # create macro in reverse
