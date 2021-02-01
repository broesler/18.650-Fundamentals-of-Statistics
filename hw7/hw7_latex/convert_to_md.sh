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

post_name="ks2samp"
main_texfile="hw7_main.tex"
# main_outfile="${main_texfile/.*/}.md"
post_texfile="$post_name.tex"
post_outfile="$post_name.md"

figure_path="/assets/images/$post_name/"

WHITE_SQUARE=$'\u25FB'  # hex code from `echo ◻ | hexdump -C`

# TODO Slice K-S Test section BEFORE filtering so we get the full preamble, and
# then figure numbers, etc. will be translated correctly.
sed -E -n -e '1,/\\maketitle/ p' \
          -e '/^\\section\{Kolmogorov/,/^\\section/ { /^\\section\{[^K]/! p}' \
          -e '/\\end\{document\}/p' \
          "$main_texfile" > "$post_texfile"

# Filter some LaTeX before using pandoc, then filter the markdown.
sed -E -f before.sed "$post_texfile" \
    | pandoc -f latex -t gfm+tex_math_dollars+footnotes --mathjax \
    | sed -E '/\\displaystyle/ {:x; N; /EndFor/b; s/\n */ /g; b x}' \
    | sed -E '/\\intertext/ {:x; N; /&/b; s/\n */ /g; b x}' \
    | sed -E -f after.sed \
    | sed -E -e "/^[[:blank:]]*$WHITE_SQUARE$/d" \
        -e "s/$WHITE_SQUARE/<span class=\"qed_symbol\">\0<\/span>/g" \
        -e '/\\tag/! s/\\qedhere/\\tag*{\0}/' \
        -e "/qedhere/ s/\\\qedhere/$WHITE_SQUARE/" \
    | cat -s \
    > "$post_outfile"

# Update figure paths, use "img" instead of "embed", fix numbers
sed -E -i'' "s,<embed\s+src=\"([^\"]*)\",<img src=\"${figure_path}\1\",g" "$post_outfile"
sed -E -i'' '/Figure[[:blank:]]<a href=\"#fig/ {
                s/Figure[[:blank:]]<a/<a/g;
                s@>([0-9]+)<@>Figure \1<@g
             }' "$post_outfile"

awk -i inplace \
    'BEGIN {count=1} 
    /<figcaption/ {
        sub(/<figcaption[^>]*>/, "<figcaption><span class=\"fig_number\">Figure " count "</span>. ");
        count++
    };
    // {print $0}' \
    "$post_outfile"

## TODO Number equations!? replace "\numberthis"

# Extract post title
the_title=$(sed -E -n '1s/^# (.*)/\1/p' "$post_outfile")
sed -i'' '1d' "$post_outfile"   # remove header line

# Prepend preamble
# <https://stackoverflow.com/questions/55435352/bad-file-descriptor-when-reading-from-fd-3-pointing-to-a-temp-file>
tmpfile=$(mktemp /tmp/$post_name.XXXXX)
exec 3>"$tmpfile"  # file descriptor to write
exec 4<"$tmpfile"  # file descriptor to read
rm "$tmpfile"      # remove the file when the script exits

cat "$post_outfile" >&3  # write the output file to a temp file

the_date='2021-01-27'

# Write the preamble to the post_outfile
cat > "$post_outfile" << EOF
---
layout: post
title:  "$the_title"
date: $the_date
categories: statistics
tags: statistics hypothesis-testing python
---

\$\$
\newcommand{\coloneqq}{\mathrel{\vcenter{:}}=}
\newcommand{\ceil}[1]{\left\lceil #1 \right\rceil}
\newcommand{\floor}[1]{\left\lfloor #1 \right\rfloor}
\$\$

EOF

# date:   "$(date +'%F %T %z')"

# Write the actual markdown to the post_outfile
cat <&4 >> "$post_outfile"

# Copy it to the final post name
# cp -i "$post_outfile" "$HOME/src/web_dev/broesler.github.io/_drafts/$(date +'%F')-$post_outfile"
cp -i "$post_outfile" "$HOME/src/web_dev/broesler.github.io/_drafts/$the_date-$post_outfile"

#===============================================================================
#===============================================================================
