#!/usr/local/bin/bash
#===============================================================================
#     File: convert_to_md.sh
#  Created: 2021-01-22 22:15
#   Author: Bernie Roesler
#
#  Description: 
#
#===============================================================================

# NOTE: 
#   * use github-flavored-markedown for Jekyll
#   * convert un-MathJax phrasing first, then pandoc it

# if [ $# -eq 0 ]; then
#     printf "Usage: ./convert_to_md.sh <filename.tex>\n" 1>&2
#     exit 1
# fi

texfile="hw7_main.tex"
main_outfile="${texfile/.*/}.md"
outfile="ks2samp.md"

sed -E 's/\\m(left|right)/\\\1/g' "$texfile" \
    | pandoc -f latex -t gfm+tex_math_dollars "$@" \
    | sed '/``` /,/# <<begin/{/``` /!d};/# <<end/,/```/{/```/!d}' \
    > "$main_outfile"

# Slice out K-S Test section to new file
sed -E -n '/^# Kolmogorov/,/^# .*/{/^# [^K]/!p}' < "$main_outfile" > "$outfile"

# Update figure paths
figure_path='/assets/images/ks2samp/'
sed -E -i'' "s,(<embed\s+src=)\"([^\"]*)\",\1\"${figure_path}\2\",g" "$outfile"

# Extract post title
the_title=$(sed -E -n 's/^# (.*)/\1/p' "$outfile")
the_date=$()

# Prepend preamble
# <https://stackoverflow.com/questions/55435352/bad-file-descriptor-when-reading-from-fd-3-pointing-to-a-temp-file>
tmpfile=$(mktemp /tmp/ks2samp.XXXXX)
exec 3>"$tmpfile"  # file descriptor to write
exec 4<"$tmpfile"  # file descriptor to read
rm "$tmpfile"      # remove the file when the script exits

cat "$outfile" >&3  # write the output file to a temp file

# Write the preamble to the outfile
cat > "$outfile" << EOF
---
layout: post
title:  "${the_title}"
date:   "$(date +'%F %T %z')"
categories: statistics
tags: statistics hypothesis-testing python
---
EOF

# Write the actual markdown to the outfile
cat <&4 >> "$outfile"

# Copy it to the final post name
cp -i "$outfile" "$HOME/src/web_dev/broesler.github.io/_drafts/$(date +'%F')-$outfile"

#===============================================================================
#===============================================================================
