#include "stdio.h"
#include "stdlib.h"
#include "HZK32.H"
#include "ASC32.H"
#include "ast_rawosd.h"


#define     ASCII_WIDTH         (8u)        /* ASCII 宽度 */
#define     HZ_WIDTH            (16u)       /* 汉字宽度 */
#define     FRONT_HEIGHT        (16u)       /* 字符高度 */
#define     ASCII_NUM           (128u)      /* ASCII 字符个数 */
#define     HZK16_FRONT_SIZE    (267616u)   /* 汉字字库大小 */
#define     ASCII8_FRONT_SIZE   (4096u)

unsigned char * HzFrontDot = NULL ;
unsigned char *  AsciiFrontDot = NULL ;

/* 点阵查找表 */
unsigned char   dotTableNormal_u8[256][8];        /* 正常点阵 */
unsigned short  dotTableNormal_u16[256][8];      /* 正常点阵 */
unsigned short  dotTableReverse_u16[256][8];     /* 左右镜像 */

int nInitOsd = 0;

/**
 * @function:   osd_Init
 * @brief:      填充 dot table
 * @param[in]:  void
 * @param[out]: None
 * @return:     void
 */
void osd_Init(void)
{
    unsigned short  i = 0;
    unsigned char   tabColorMask_u8 = 0xFFu;
    unsigned short  tabColorMask = 0xFFFFu;
    for(i = 0; i < 256; i++)
    {
        dotTableNormal_u8[i][0] = (i & 0x80u) ? tabColorMask_u8 : 0x00u;
        dotTableNormal_u8[i][1] = (i & 0x40u) ? tabColorMask_u8 : 0x00u;
        dotTableNormal_u8[i][2] = (i & 0x20u) ? tabColorMask_u8 : 0x00u;
        dotTableNormal_u8[i][3] = (i & 0x10u) ? tabColorMask_u8 : 0x00u;
        dotTableNormal_u8[i][4] = (i & 0x08u) ? tabColorMask_u8 : 0x00u;
        dotTableNormal_u8[i][5] = (i & 0x04u) ? tabColorMask_u8 : 0x00u;
        dotTableNormal_u8[i][6] = (i & 0x02u) ? tabColorMask_u8 : 0x00u;
        dotTableNormal_u8[i][7] = (i & 0x01u) ? tabColorMask_u8 : 0x00u;

        dotTableNormal_u16[i][0] = (i & 0x80) ? tabColorMask : 0x0000;
        dotTableNormal_u16[i][1] = (i & 0x40) ? tabColorMask : 0x0000;
        dotTableNormal_u16[i][2] = (i & 0x20) ? tabColorMask : 0x0000;
        dotTableNormal_u16[i][3] = (i & 0x10) ? tabColorMask : 0x0000;
        dotTableNormal_u16[i][4] = (i & 0x08) ? tabColorMask : 0x0000;
        dotTableNormal_u16[i][5] = (i & 0x04) ? tabColorMask : 0x0000;
        dotTableNormal_u16[i][6] = (i & 0x02) ? tabColorMask : 0x0000;
        dotTableNormal_u16[i][7] = (i & 0x01) ? tabColorMask : 0x0000;

        dotTableReverse_u16[i][7] = (i & 0x80) ? tabColorMask : 0x0000;
        dotTableReverse_u16[i][6] = (i & 0x40) ? tabColorMask : 0x0000;
        dotTableReverse_u16[i][5] = (i & 0x20) ? tabColorMask : 0x0000;
        dotTableReverse_u16[i][4] = (i & 0x10) ? tabColorMask : 0x0000;
        dotTableReverse_u16[i][3] = (i & 0x08) ? tabColorMask : 0x0000;
        dotTableReverse_u16[i][2] = (i & 0x04) ? tabColorMask : 0x0000;
        dotTableReverse_u16[i][1] = (i & 0x02) ? tabColorMask : 0x0000;
        dotTableReverse_u16[i][0] = (i & 0x01) ? tabColorMask : 0x0000;
    }

    HzFrontDot = hzk_buf ;
    AsciiFrontDot = asc_buf ;

    return;
}


/**
 * @function:   getCharFrontAddr
 * @brief:      获取汉字字库地址
 * @param[in]:  unsigned short  charHz
 * @param[out]: None
 * @return:     unsigned char   *
 */
unsigned char   * getCharFrontAddr(unsigned short  charCode)
{
    if(charCode < 0xFF)
    {
        return (AsciiFrontDot + charCode * 16);
    }
    else
    {
        unsigned int  offset = (((charCode & 0xFFu) - 0xA1u) + 94u * ((charCode >> 8u) - 0xA1u) ) << 5;
        return (HzFrontDot + offset);
    }
}




/**
 * @function:   cpu_DrawAscii
 * @brief:      cpu 绘制 Ascii 点阵
 * @param[in]:  unsigned char   * pFont     字符字库地址
 * @param[in]:  unsigned short  * pDst     目的点阵地址(需手动计算起始像素地址)
 * @param[in]:  uint32_t pitch      行 pitch (单位像素, 2B)
 * @param[in]:  unsigned char   scale       字符规模(8x16 的倍数)
 * @param[in]:  unsigned short  color      像素颜色(ARGB 2byte)
 * @param[out]: None
 * @return:     void
 */
void cpu_DrawAscii(unsigned char   *   pFont,
	unsigned short  *  pDst,
	int     pitch,
	unsigned char       scale,
	unsigned short     color)
{
	if ((pFont == NULL) || (pDst == NULL))
	{

		return;
	}

	unsigned char   hIdx, k, j;
	unsigned short  * pDotTableValue = NULL;
	unsigned long long  * pTemp64 = (unsigned long long  *)pDst;  /* 用 64bit 加速行搬移*/
	unsigned short   *pTemp = (unsigned short  *)pDst;


    //unsigned long long
#if 1   /* 展开部分循环后, 性能可提升 1/3 */
	switch (scale)
	{
	case 1:
		for (hIdx = 0; hIdx < FRONT_HEIGHT; hIdx++)
		{
			pDotTableValue = dotTableNormal_u16[pFont[hIdx]];
			for (k = 0; k < ASCII_WIDTH; k++)
			{
				pTemp[hIdx * pitch + k] = pDotTableValue[k] & color;
			}
		}
		break;
	case 2:
		for (hIdx = 0; hIdx < FRONT_HEIGHT; hIdx++)
		{
			pDotTableValue = dotTableNormal_u16[pFont[hIdx]];
			for (k = 0; k < ASCII_WIDTH; k++)
			{
				pTemp[(hIdx * pitch + k) * 2 + 0] = pDotTableValue[k] & color;
				pTemp[(hIdx * pitch + k) * 2 + 1] = pDotTableValue[k] & color;
			}
		}
		break;
	case 3:
		for (hIdx = 0; hIdx < FRONT_HEIGHT; hIdx++)
		{
			pDotTableValue = dotTableNormal_u16[pFont[hIdx]];
			for (k = 0; k < ASCII_WIDTH; k++)
			{
				pTemp[(hIdx * pitch + k) * 3 + 0] = pDotTableValue[k] & color;
				pTemp[(hIdx * pitch + k) * 3 + 1] = pDotTableValue[k] & color;
				pTemp[(hIdx * pitch + k) * 3 + 2] = pDotTableValue[k] & color;
			}
		}
		break;
	case 4:
		for (hIdx = 0; hIdx < FRONT_HEIGHT; hIdx++)
		{
			pDotTableValue = dotTableNormal_u16[pFont[hIdx]];
			for (k = 0; k < ASCII_WIDTH; k++)
			{
				pTemp[(hIdx * pitch + k) * 4 + 0] = pDotTableValue[k] & color;
				pTemp[(hIdx * pitch + k) * 4 + 1] = pDotTableValue[k] & color;
				pTemp[(hIdx * pitch + k) * 4 + 2] = pDotTableValue[k] & color;
				pTemp[(hIdx * pitch + k) * 4 + 3] = pDotTableValue[k] & color;
			}
		}
		break;
	default:
		break;
	}

	for (hIdx = 0; hIdx < FRONT_HEIGHT; hIdx++)
	{
		/* 扩展剩余的行(复制已经生成的行点阵数据) */
		for (k = 1; k < scale; k++)
		{
			/* 每行需要搬移的数据量为 (ASCII_WIDTH * 2 / 8) */
			for (j = 0; j < (ASCII_WIDTH * 2 / 8 * scale); j++)
			{
				pTemp64[(scale * hIdx + k) * pitch / 4 + j] = pTemp64[scale * hIdx * pitch / 4 + j];
			}
		}
	}
#else
	for (hIdx = 0; hIdx < FRONT_HEIGHT; hIdx++)
	{
		pDotTableValue = dotTableNormal_u16[pFont[hIdx]];

		/* 生成字符的行 */
		for (k = 0; k < ASCII_WIDTH; k++)
		{
			/* 扩展字符的行宽度 */
			for (j = 0; j < scale; j++)
			{
				pTemp[(hIdx * pitch + k) * scale + j] = pDotTableValue[k] & color;
			}
		}

		/* 扩展剩余的行(复制已经生成的行点阵数据) */
		for (k = 1; k < scale; k++)
		{
			/* 每行需要搬移的数据量为 (ASCII_WIDTH * 2 / 8) */
			for (j = 0; j < (ASCII_WIDTH * 2 / 8 * scale); j++)
			{
				pTemp64[(scale * hIdx + k) * pitch / 4 + j] = pTemp64[scale * hIdx * pitch / 4 + j];
			}
		}
	}
#endif

	return;
}


/**
 * @function:   cpu_DrawAscii
 * @brief:      cpu 绘制 汉字点阵
 * @param[in]:  unsigned char   * pFont     字符字库地址
 * @param[in]:  unsigned short  * pDst     目的点阵地址(需手动计算起始像素地址)
 * @param[in]:  uint32_t pitch      行 pitch (单位像素, 2B)
 * @param[in]:  unsigned char   scale       字符规模(8x16 的倍数)
 * @param[in]:  unsigned short  color      像素颜色(ARGB 2byte)
 * @param[out]: None
 * @return:     void
 */
void cpu_DrawChinese(unsigned char   *   pFont,
	unsigned short  *  pDst,
	int    pitch,
	unsigned char       scale,
	unsigned short     color)
{
	if ((pFont == NULL) || (pDst == NULL))
	{

		return;
	}

	unsigned char   hIdx, k, j;
	unsigned short  * pDotTableValue_1 = NULL;
	unsigned short  * pDotTableValue_2 = NULL;     /*汉字基础宽度为 16 像素, 字库中为 2Byte */
	unsigned short   *pTemp = (unsigned short  *)pDst;
	unsigned long long  * pTemp64 = (unsigned long long  *)pDst;

#if 1   /* 展开部分循环后, 性能可提升 1/3 */
	switch (scale)
	{
	case 1:
		for (hIdx = 0; hIdx < FRONT_HEIGHT; hIdx++)
		{
			pDotTableValue_1 = dotTableNormal_u16[pFont[hIdx * 2]];
			pDotTableValue_2 = dotTableNormal_u16[pFont[hIdx * 2 + 1]];
			for (k = 0; k < ASCII_WIDTH; k++)
			{
				pTemp[hIdx * pitch + k] = pDotTableValue_1[k] & color;
				pTemp[hIdx * pitch + k + ASCII_WIDTH] = pDotTableValue_2[k] & color;
			}
		}
		break;
	case 2:
		for (hIdx = 0; hIdx < FRONT_HEIGHT; hIdx++)
		{
			pDotTableValue_1 = dotTableNormal_u16[pFont[hIdx * 2]];
			pDotTableValue_2 = dotTableNormal_u16[pFont[hIdx * 2 + 1]];
			for (k = 0; k < ASCII_WIDTH; k++)
			{
				pTemp[(hIdx * pitch + k) * 2 + 0] = pDotTableValue_1[k] & color;
				pTemp[(hIdx * pitch + k) * 2 + 1] = pDotTableValue_1[k] & color;
				pTemp[(hIdx * pitch + ASCII_WIDTH + k) * 2 + 0] = pDotTableValue_2[k] & color;
				pTemp[(hIdx * pitch + ASCII_WIDTH + k) * 2 + 1] = pDotTableValue_2[k] & color;
			}
		}
		break;
	case 3:
		for (hIdx = 0; hIdx < FRONT_HEIGHT; hIdx++)
		{
			pDotTableValue_1 = dotTableNormal_u16[pFont[hIdx * 2]];
			pDotTableValue_2 = dotTableNormal_u16[pFont[hIdx * 2 + 1]];
			for (k = 0; k < ASCII_WIDTH; k++)
			{
				pTemp[(hIdx * pitch + k) * 3 + 0] = pDotTableValue_1[k] & color;
				pTemp[(hIdx * pitch + k) * 3 + 1] = pDotTableValue_1[k] & color;
				pTemp[(hIdx * pitch + k) * 3 + 2] = pDotTableValue_1[k] & color;
				pTemp[(hIdx * pitch + ASCII_WIDTH + k) * 3 + 0] = pDotTableValue_2[k] & color;
				pTemp[(hIdx * pitch + ASCII_WIDTH + k) * 3 + 1] = pDotTableValue_2[k] & color;
				pTemp[(hIdx * pitch + ASCII_WIDTH + k) * 3 + 2] = pDotTableValue_2[k] & color;
			}
		}
		break;
	case 4:
		for (hIdx = 0; hIdx < FRONT_HEIGHT; hIdx++)
		{
			pDotTableValue_1 = dotTableNormal_u16[pFont[hIdx * 2]];
			pDotTableValue_2 = dotTableNormal_u16[pFont[hIdx * 2 + 1]];
			for (k = 0; k < ASCII_WIDTH; k++)
			{
				pTemp[(hIdx * pitch + k) * 4 + 0] = pDotTableValue_1[k] & color;
				pTemp[(hIdx * pitch + k) * 4 + 1] = pDotTableValue_1[k] & color;
				pTemp[(hIdx * pitch + k) * 4 + 2] = pDotTableValue_1[k] & color;
				pTemp[(hIdx * pitch + k) * 4 + 3] = pDotTableValue_1[k] & color;
				pTemp[(hIdx * pitch + ASCII_WIDTH + k) * 4 + 0] = pDotTableValue_2[k] & color;
				pTemp[(hIdx * pitch + ASCII_WIDTH + k) * 4 + 1] = pDotTableValue_2[k] & color;
				pTemp[(hIdx * pitch + ASCII_WIDTH + k) * 4 + 2] = pDotTableValue_2[k] & color;
				pTemp[(hIdx * pitch + ASCII_WIDTH + k) * 4 + 3] = pDotTableValue_2[k] & color;
			}
		}
		break;
	default:
		break;
	}

	for (hIdx = 0; hIdx < FRONT_HEIGHT; hIdx++)
	{
		for (k = 1; k < scale; k++)
		{
			for (j = 0; j < (HZ_WIDTH * 2 / 8 * scale); j++)
			{
				pTemp64[(scale * hIdx + k) * pitch / 4 + j] = pTemp64[scale * hIdx * pitch / 4 + j];
			}
		}
	}
#else
	for (hIdx = 0; hIdx < FRONT_HEIGHT; hIdx++)
	{
		pDotTableValue_1 = dotTableNormal_u16[pFont[hIdx * 2]];
		pDotTableValue_2 = dotTableNormal_u16[pFont[hIdx * 2 + 1]];

		for (k = 0; k < ASCII_WIDTH; k++)
		{
			for (j = 0; j < scale; j++)
			{
				pTemp[(hIdx * pitch + k) * scale + j] = pDotTableValue_1[k] & color;
				pTemp[(hIdx * pitch + k + ASCII_WIDTH) * scale + j] = pDotTableValue_2[k] & color;
			}
		}

		for (k = 1; k < scale; k++)
		{
			for (j = 0; j < (HZ_WIDTH * 2 / 8 * scale); j++)
			{
				pTemp64[(scale * hIdx + k) * pitch / 4 + j] = pTemp64[scale * hIdx * pitch / 4 + j];
			}
		}
	}

#endif

	return;
}



static  int draw_text(void * pMem, unsigned char   scale, char *testOsdString)
{
	int stringLen = strlen(testOsdString);
	int i = 0;
	char * pChar = NULL;
	uint32_t pitch = 0;
	unsigned short  * charDstStartX = NULL;
	char *pt = testOsdString;
	charDstStartX = (unsigned short *)pMem;
	unsigned char   * pCharAddr = NULL;
	unsigned char   incode[2] = { 0, 0 };

	int pictchs = 0;
	int  hanzicount = 0;
	int  asciicount = 0;

	pt = testOsdString;
	while ((*pt) != 0)
	{
		incode[0] = (unsigned char)* pt;
		incode[1] = (unsigned char) * (pt + 1);
		if ((int)incode[0] >> 7)//chinease character
		{
			hanzicount++;
			pt += 2;
		}
		else {
			pt += 1;
			asciicount++;
		}
	}
	pt = testOsdString;

	pictchs = hanzicount * HZ_WIDTH * scale + asciicount * ASCII_WIDTH * scale;


	i = 0;

	unsigned short * pFont = (unsigned short *)pMem;

	unsigned short *charDstStartX_Hz = pFont;

	while ((*pt) != 0)
	{
		incode[0] = (unsigned char)* pt;
		incode[1] = (unsigned char) * (pt + 1);
		if ((int)incode[0] >> 7)//chinease character
		{
			unsigned short  HzValue = (unsigned short )((incode[0] << 8u) | incode[1]);
			unsigned char   * pCharAddr = getCharFrontAddr(HzValue);
			charDstStartX_Hz = pFont;
			pt += 2;

			cpu_DrawChinese(pCharAddr, charDstStartX_Hz, pictchs, scale,0xFFFF);
			pFont += (HZ_WIDTH * scale);
			i++;

		}
		else
		{
			unsigned char   * pCharAddr = NULL;
			char * pChar = NULL;
			pCharAddr = getCharFrontAddr(incode[0]);
			unsigned short  * charDstStartX = pFont;

			cpu_DrawAscii(pCharAddr, charDstStartX, pictchs, scale ,0xFFFF);
			i++;
			pt++;
			pFont += (ASCII_WIDTH * scale);
		}
	}


	return pictchs;
}


int   dt_raw_text(unsigned char *pImg, int nwidth, int nheight, int x, int y, char *text, int textlen, int nscale )
{

	char *pt = 0;	//point to the character you want to put
	char *pt_c = 0;	//point to the character lib we are checking

	unsigned short  pDot[128 * 1024] = { 0 };
	
	if(nInitOsd ==0 )
	{
		nInitOsd=1;
		osd_Init () ;
	}

	if(x <0 || y<0 || x > nwidth - 7*32 ||y > nheight -32 ) return  0;


	int osdw = draw_text(pDot,nscale ,text);


	int osdh = FRONT_HEIGHT * nscale;
	int widstep = nwidth * 3;
	unsigned char *pOsdOff = pImg + y * (widstep)+x*3;


	int nPix = 255;
	int nPix1 = 0;


	unsigned short  lbuf[128 * 1024] = { 0 };

	memcpy(lbuf, pDot, 128 * 1024);

	unsigned short *lpix = (unsigned short *)lbuf;
	int i, j;

	for (i = 0; i < osdh; i++)
	{
		for (j = 0; j < osdw - 1; j++)
		{
			if (lpix[i*osdw + j] ^ lpix[i*osdw + j + 1])
			{
				if (lpix[i*osdw + j] == 0)
				{
					lpix[i*osdw + j] = 0xEEEE;
				}
				else
				{
					lpix[i*osdw + j + 1] = 0xEEEE;
					j++;
				}
			}
		}
	}


	unsigned short  vbuf[128 * 1024] = { 0 };

	memcpy(vbuf, pDot, 128 * 1024);

	unsigned short *vpix = (unsigned short *)vbuf;

	for (i = 0; i < osdw; i++)
	{
		for (j = 0; j < osdh - 1; j++)
		{
			if (vpix[j*osdw + i] ^ vpix[(j + 1)*osdw + i])
			{
				if (vpix[j*osdw + i] == 0)
				{
					vpix[j*osdw + i] = 0XEEEE;
				}
				else
				{
					vpix[(j + 1)*osdw + i] = 0XEEEE;
					j++;
				}
			}
		}
	}

	//mixture;
	unsigned short *pData = (unsigned short *)pDot;

	for (i = 0; i < osdh; i++)
	{
		for (j = 0; j <osdw; j++)
		{
			pData[i*osdw + j] = lpix[i*osdw + j];
			pData[i*osdw + j] |= vpix[i*osdw + j];
		}
	}



	for (int i = 0; i < osdh; i++)
	{

		pOsdOff = pImg + (y+i) * (widstep)+x*3;
		for (int j = 0; j < osdw; j++)
		{
			if (pDot[i*osdw + j] == 0xFFFF)
			{
				*pOsdOff++ = nPix;
				*pOsdOff++ = nPix;
				*pOsdOff++ = nPix;
			}
			else   if (pDot[i*osdw + j] == 0xEEEE)
			{
				*pOsdOff++ = nPix1;
				*pOsdOff++ = nPix1;
				*pOsdOff++ = nPix1;
			}
			else
			{
				 pOsdOff += 3;
			}
		}
	}

	return 0;

}






