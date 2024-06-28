run-pull-force:
	@if [ -n "$$(docker images -f 'dangling=true' -q)" ]; then \
			docker rmi $$(docker images -f 'dangling=true' -q); \
		else \
			echo "No dangling images to remove."; \
			exit 0; \
		fi	