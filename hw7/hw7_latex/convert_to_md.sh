#!/usr/local/bin/bash
#===============================================================================
#     File: convert_to_md.sh
#  Created: 2021-01-22 22:15
#   Author: Bernie Roesler
#
#  Description: 
#
#===============================================================================

# TODO include argument parser to generalize script
# if [ $# -eq 0 ]; then
#     printf "Usage: ./convert_to_md.sh <filename.tex>\n" 1>&2
#     exit 1
# fi

post_basename="ks2samp"
texfile="hw7_main.tex"
main_outfile="${texfile/.*/}.md"
outfile="$post_basename.md"

# white_square='\xE2\x97\xBB'  # hex code from `echo ◻ | hexdump -C`

# TODO figure out how to use pandoc template directly?
# Filter some LaTeX before using pandoc, then filter the markdown.
sed -E -f before.sed "$texfile" \
    | pandoc --mathjax -f latex -t gfm+tex_math_dollars+footnotes \
    | sed -E -f after.sed \
    | cat -s \
    > "$main_outfile"

# Slice out K-S Test section to new file
sed -E -n '/^# Kolmogorov/,/^# .*/{/^# [^K]/!p}' "$main_outfile" > "$outfile"

# Update figure paths, use "img" instead of "embed"
figure_path="/assets/images/$post_basename/"
sed -E -i'' "s,<embed\s+src=\"([^\"]*)\",<img src=\"${figure_path}\1\",g" "$outfile"

# Extract post title
the_title=$(sed -E -n '1s/^# (.*)/\1/p' "$outfile")
sed -i'' '1d' "$outfile"   # remove header line

# Prepend preamble
# <https://stackoverflow.com/questions/55435352/bad-file-descriptor-when-reading-from-fd-3-pointing-to-a-temp-file>
tmpfile=$(mktemp /tmp/$post_basename.XXXXX)
exec 3>"$tmpfile"  # file descriptor to write
exec 4<"$tmpfile"  # file descriptor to read
rm "$tmpfile"      # remove the file when the script exits

cat "$outfile" >&3  # write the output file to a temp file

the_date=$(date +%F)

# Write the preamble to the outfile
cat > "$outfile" << EOF
---
layout: post
title:  "$the_title"
date: $the_date
categories: statistics
tags: statistics hypothesis-testing python
---

\$\$
\newcommand{\coloneqq}{\mathrel{\vcenter{:}}=}
\newcommand{\indic}[1]{\unicode[Garamond]{x1D7D9}\!\left\{ #1 \right\}}
\newcommand{\ceil}[1]{\left\lceil #1 \right\rceil}
\newcommand{\floor}[1]{\left\lfloor #1 \right\rfloor}
\$\$

EOF

# date:   "$(date +'%F %T %z')"

# Write the actual markdown to the outfile
cat <&4 >> "$outfile"

# Copy it to the final post name
# cp -i "$outfile" "$HOME/src/web_dev/broesler.github.io/_drafts/$(date +'%F')-$outfile"
# cp -i "$outfile" "$HOME/src/web_dev/broesler.github.io/_drafts/$the_date-$outfile"

##===============================================================================
##===============================================================================
