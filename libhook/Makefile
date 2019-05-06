all: hook.c
	g++ -o libhook.so -fPIC -shared hook.c -ldl -I$(ALTERAOCLSDKROOT)/board/nalla_pcie/software/include

install: libhook.so
	cp libhook.so $(HOME)/root/lib

clean:
	rm -rf libhook.so
