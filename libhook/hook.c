#include <sys/types.h>
#include <stdio.h>
#include <string.h>
#include <aocl_mmd.h>

#include <dlfcn.h>

/* define a type of function pointer */
typedef int (*aocl_mmd_open_t) (const char *);

int aclnalla_pcie0_handle = -1;

/* http://optumsoft.com/dangers-of-using-dlsym-with-rtld_next/ is an interesting
 * introduction to the topic.  As mentioned in the post, the idea is to have
 * a function wrapper for the real aocl_mmd_open function. 
 */ 
AOCL_MMD_CALL int aocl_mmd_open(const char *name)
{
	static aocl_mmd_open_t aocl_mmd_open_real  = NULL;
	if (!aocl_mmd_open_real) {
    // find the address of the function "aocl_mmd_open"
		aocl_mmd_open_real = (aocl_mmd_open_t) dlsym(RTLD_NEXT, "aocl_mmd_open");
	}
	if (strcmp("aclnalla_pcie0", name) == 0) {
		aclnalla_pcie0_handle = aocl_mmd_open_real(name);
		printf("kaicheng: aclnalla_pcie0_handle = %d.\n", aclnalla_pcie0_handle);
		return aclnalla_pcie0_handle;
	}
	return aocl_mmd_open_real(name);
} 
