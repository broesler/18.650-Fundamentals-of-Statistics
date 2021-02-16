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

# TODO 
#   * add expected reading time!

# post_name="ks2samp"
post_name="indep_bern"
main_texfile="hw7_main.tex"

categories="statistics"
tags="statistics hypothesis-testing python"

post_texfile="$post_name.tex"
post_outfile="$post_name.md"

figure_path="/assets/images/$post_name/"

# White squares used for proof endings except for "wrapper" proofs, which use
# black squares.
WHITE_SQUARE=$'\u25FB'  # hex code from `echo ◻ | hexdump -C`
BLACK_SQUARE=$'\u25A0'  # hex code from `echo ■ | hexdump -C`

# TODO convert pattern to $pattern and ${pattern:0:1}
# Slice K-S Test section BEFORE pandoc so we get the full preamble, and
# then figure numbers, etc. will be translated correctly.
# -e '/^\\section\{Kolmogorov/,/^\\section/ { /^\\section\{[^K]/! p}' \
sed -E -n -e '1,/\\maketitle/ p' \
          -e '/^\\section\{Aside/,/^\\section/ { /^\\section\{[^A]/! p}' \
          -e '/\\end\{document\}/p' \
          "$main_texfile" > "$post_texfile"

# NOTE 
#   * line-gathering sed calls should be separated
#   * commands using bash variables need to be here, not in `(before|after).sed`

# Filter some LaTeX before using pandoc, then filter the markdown.
sed -E -f before.sed "$post_texfile" \
    | pandoc -f latex -t gfm+tex_math_dollars+footnotes \
        --mathjax --filter pandoc-sidenote \
    | sed -E '/\\displaystyle/ {:x; N; s/\n */ /g; /\}\$/b; b x}' \
    | sed -E '/\\intertext/ {:x; N; /&/b; s/\n */ /g; b x}' \
    | sed -E -f after.sed \
    | sed -E -e "/^[[:blank:]]*$WHITE_SQUARE$/d" \
        -e "s/$WHITE_SQUARE/<span class=\"qed_symbol\">\0<\/span>/g" \
        -e '/\\tag/! s/\\qedhere/\\tag*{\0}/' \
        -e "/\\\qedhere/ s//$WHITE_SQUARE/" \
        -e "/\\\qed/ s//$BLACK_SQUARE/" \
        -e "s,<embed\s+src=\"([^\"]*)\",<img src=\"{{ '${figure_path}\1' | absolute_url }}\",g" \
    | sed -E \
        '/\\begin\{align/,/\\end\{align/ { 
            /\\begin\{(bmatrix|split)\}/,/\\end\{(bmatrix|split)\}/! {
                /\\(begin|end)/! {
                    /\\numberthis/! {
                        /\\tag/! {
                            s/(\\\\)?$/\\nonumber\1/g
                        }
                    }
                }
            }
            /\\end\{(bmatrix|split)\}/ s/(\\\\)?$/\\nonumber\1/g
        }' \
    | sed -E 's/\\numberthis//' \
    | cat -s \
    > "$post_outfile"

# Number figures and algorithms.
# Captions: '(Figure|Algorithm) X. [the caption here]'.
awk -i inplace \
    'BEGIN {
        fig_count=1;
        alg_count=1;
    } 
    /<figcaption/ {
        sub(/<figcaption[^>]*>/, "<figcaption><span class=\"fig_number\">Figure " fig_count "</span>. ");
        fig_count++;
    };
    /<span class="alg_title">/ {
        sub(/<span class="alg_title">Algorithm/, "<span class=\"alg_title\">Algorithm " alg_count ". ");
        alg_count++;
    };
    {print $0}' \
    "$post_outfile"

# Subsitute any remaining `\ref`s
sed -E -i'' '/\\ref/ s@Algorithm~\\ref\{([^}]*)\}@Algorithm <a href="#\1">[\1]</a>@g' "$post_outfile"

# Update Algorithm numbers in links
kvs=$(sed -E -n '
    /<div class="algorithm"/ {
    :x
    N
    /<span class="alg_title"/ {
        s/.*id="([^"]*)".*Algorithm ([0-9]+)\..*/\1=\2/
        p
        b
    }
    b x
}' "$post_outfile")

while IFS='=' read -r key val; do
    sed -E -i'' "s/Algorithm[[:blank:]](<a [^>]*>)\[$key\]</\1Algorithm\&nbsp;$val</" "$post_outfile"
done < <(echo "$kvs")

# Extract post title
the_title=$(sed -E -n '1s/^# (.*)/\1/p' "$post_outfile")
sed -i'' '1d' "$post_outfile"   # remove header line

reading_time=$(./reading_time.sh "$post_outfile")

#------------------------------------------------------------------------------- 
#        Prepend preamble
#-------------------------------------------------------------------------------
# <https://stackoverflow.com/questions/55435352/bad-file-descriptor-when-reading-from-fd-3-pointing-to-a-temp-file>
tmpfile=$(mktemp /tmp/$post_name.XXXXX)
exec 3>"$tmpfile"  # file descriptor to write
exec 4<"$tmpfile"  # file descriptor to read
rm "$tmpfile"      # remove the file when the script exits

cat "$post_outfile" >&3  # write the output file to a temp file

the_date='2021-02-15'
# the_date=$(date +'%F')

# Write the preamble to the post_outfile
cat > "$post_outfile" << EOF
---
layout: post
title:  "$the_title"
date:   "$(date +'%F %T %z')"
categories: $categories
tags: $tags
reading_time: $reading_time
---

<div style="visibility: hidden; padding: 0; margin-bottom: -2rem;">
\$\$
\begin{align*}
\newcommand{\coloneqq}{\mathrel{\vcenter{:}}=}
\newcommand{\ceil}[1]{\left\lceil #1 \right\rceil}
\newcommand{\floor}[1]{\left\lfloor #1 \right\rfloor}
\newcommand{\indep}{\perp \hspace{-18mu} \perp}
\newcommand{\nindep}{\not \hspace{-13mu} \indep}
\end{align*}
\$\$
</div>

EOF

# Write the actual markdown to the post_outfile
cat <&4 >> "$post_outfile"

# Copy it to the final post name
cp -i "$post_outfile" "$HOME/src/web_dev/broesler.github.io/_drafts/$the_date-$post_outfile"

#===============================================================================
#===============================================================================
