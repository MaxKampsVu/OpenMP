all: libannotation.a

libannotation.a: annotation.o
	ar rcs libannotation.a annotation.o

annotation.o: ../src/annotation.c
	gcc -c $^

clean:
	rm -f *.a *.o
