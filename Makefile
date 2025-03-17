.PHONY: tarball

tarball: report.pdf heat_omp heat_seq include lib src
	make -C heat_seq clean
	make -C heat_omp clean
	tar -chzf heat_submit.tar.gz $^

