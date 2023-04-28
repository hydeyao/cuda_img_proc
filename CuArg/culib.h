#pragma once
#ifndef __CUDALIB_H__
#define __CUDALIB_H__

#define CU_EXPORT __declspec(dllexport)

extern "C"
{
	CU_EXPORT void cudalib_init();
	CU_EXPORT void cudalib_cleanup();
	CU_EXPORT void cudalib_process(unsigned char* src, unsigned char* dst, int width, int height);
}

#endif // __CUDALIB_H__


