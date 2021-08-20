all: main.pdf suppl.pdf

main.pdf: main.tex descr-stats-ao-events.tex descr-stats-av-events.tex descr-stats-anno.tex
	latexmk -pdf -g $<

suppl.pdf: suppl.tex
	latexmk -pdf -g $<

clean:
	latexmk -C
	rm -f main.aux main.bbl main.blg main.fb_latexmk main.fls main.log main.out
	rm -f suppl.aux suppl.bbl suppl.blg suppl.fb_latexmk suppl.fls suppl.log suppl.out

.PHONY: clean
