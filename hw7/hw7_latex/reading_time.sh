#!/usr/local/bin/bash
#===============================================================================
#     File: reading_time.sh
#  Created: 2021-02-03 16:15
#   Author: Bernie Roesler
#
#  Description: Estimate reading time of a markdown file
#
#===============================================================================

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 filename.md" 2>&1
    exit 1
fi

# Count words (not in equations or figures), equations, and figures
words=$(sed -E -n '/\$\$/,/\$\$/! { /<figure>/,/<\/figure>/! p}' "$1" | wc -w)
eqns=$(sed -E -n '/\$\$/,/\$\$/ { /\$\$/! { /\\(begin|end)/! p}}' "$1"  | wc -l)
figs=$(sed -E -n '/<img/ p' "$1"  | wc -l)

# Estimate reading time
# <https://blog.medium.com/read-time-and-you-bc2048ab620c>
WPM=275  # [words/min] avg adult human reading time
SPERFIG=12  # [sec/fig] seconds per figure 
SPEREQN=10  # [sec/eqn] seconds per equation 

reading_time=$(echo "$words / $WPM\
                    + $SPERFIG/60 * $figs\
                    + $SPEREQN/60 * $eqns" | bc -l)

# echo $reading_time
printf '%.0f\n' $reading_time

#===============================================================================
#===============================================================================
