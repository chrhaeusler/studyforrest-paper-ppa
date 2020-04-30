all: main.pdf

main.pdf: main.tex descr-stats-ao-events.tex descr-stats-av-events.tex
	latexmk -pdf -g $<

clean:
	latexmk -C
	rm -f main.aux main.bbl main.blg main.fb_latexmk main.fls main.log main.out

.PHONY: clean
