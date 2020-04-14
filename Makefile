all: paper

paper:
	$(MAKE) -C paper

clean:
	$(MAKE) -C paper clean

.PHONY: paper clean
