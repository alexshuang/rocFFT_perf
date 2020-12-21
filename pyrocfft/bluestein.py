#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from pathlib import Path


# In[2]:


path = Path('out/bluestein_111')
path.mkdir(exist_ok=True)

rocfft_precision_single = 'float'
rocfft_precision_double = 'double'

# In[3]:

def listify(v):
    if isinstance(v, list): return v
    elif isinstance(v, tuple): return list(v)
    return [v]

class Buffer():
    def __init__(self, name, nl, fp=None, offset=0, buf=None):
        self.nl = nl
        self.buf = np.arange(nl) if buf is None else buf
        self.name = name
        self.fp = fp
        self.offset = offset
    
    def __getitem__(self, i:int):
        if self.fp:
            self.fp.write(f"load {self.name}: inOffset: {self.offset + i}\n") #, value: {self.buf[i]}")
        else:
            print(f"load {self.name}: inOffset: {self.offset + i}\n")
        return self.buf[self.offset + int(i)]
    
    def __setitem__(self, i:int, val):
        if self.fp:
            self.fp.write(f"store {self.name}: outOffset: {self.offset + i}\n") #, value: {val}")
        else:
            print(f"store {self.name}: outOffset: {self.offset + i}")
        val = listify(val)
        self.buf[self.offset + int(i)] = val[0]

    def sub_array(self, name=None, offset=None):
        return Buffer(name if name is not None else self.name, self.nl, fp=self.fp, offset=offset if offset is not None else self.offset, buf=self.buf)


class RegisterBuffer():
    def __init__(self, prefix, nl, suffix=None, fp=None):
        self.buf = np.arange(nl)
        self.prefix = prefix
        self.suffix = suffix
        self.fp = fp
    
    def __getitem__(self, i:int):
        name = f"{self.prefix}{i}.{self.suffix}" if self.suffix else f"{self.prefix}{i}"
        if self.fp:
            self.fp.write(f"load {name}\n")
        else:
            print(f"load {name}") #, value: {self.buf[i]}")
        return self.buf[int(i)]
    
    def __setitem__(self, i:int, val):
        name = f"{self.prefix}{i}.{self.suffix}" if self.suffix else f"{self.prefix}{i}"
        if self.fp:
            self.fp.write(f"store {name}\n")
        else:
            print(f"store {name}") #, value: {val}")
        self.buf[int(i)] = val


# In[4]:


length = [111]
large1D = length[0] * 2
lengthBlue = 256
direction = -1
batch = batch_count = 1
precision = rocfft_precision_single

# 139         X         = size_t(1) << TWIDDLE_DEE; // 2*8 = 256
# 140         Y         = DivRoundingUp<size_t>(CeilPo2(N), TWIDDLE_DEE);
# 141         tableSize = X * Y;
tw_size = 256

outfile = open(path/'kernel_run_log.txt', 'w')
input_buf = Buffer('bufIn', lengthBlue*4, fp=outfile)
output_buf = Buffer('bufOut', lengthBlue*4, fp=outfile)
twiddles_large = Buffer('twiddle table large', tw_size, fp=outfile)

length, large1D, lengthBlue, tw_size


# In[7]:


def TWLstep1(twiddles, u):
    j      = u & 255;
    result = twiddles[j];
    return result;

def TWLstep2(twiddles, u):
    j      = u & 255;
    result = twiddles[j];
    u >>= 8;
    j      = u & 255;
    result = ((result.x * twiddles[256 + j].x - result.y * twiddles[256 + j].y),
                                 (result.y * twiddles[256 + j].x + result.x * twiddles[256 + j].y));
    return result;

def TWLstep3(twiddles, u):
    j      = u & 255;
    result = twiddles[j];
    u >>= 8;
    j      = u & 255;
    result = ((result.x * twiddles[256 + j].x - result.y * twiddles[256 + j].y),
                                 (result.y * twiddles[256 + j].x + result.x * twiddles[256 + j].y));
    u >>= 8;
    j      = u & 255;
    result = ((result.x * twiddles[512 + j].x - result.y * twiddles[512 + j].y),
                                 (result.y * twiddles[512 + j].x + result.x * twiddles[512 + j].y));
    return result;

def TWLstep4(twiddles, u):
    j      = u & 255;
    result = twiddles[j];
    u >>= 8;
    j      = u & 255;
    result = ((result.x * twiddles[256 + j].x - result.y * twiddles[256 + j].y),
                                 (result.y * twiddles[256 + j].x + result.x * twiddles[256 + j].y));
    u >>= 8;
    j      = u & 255;
    result = ((result.x * twiddles[512 + j].x - result.y * twiddles[512 + j].y),
                                 (result.y * twiddles[512 + j].x + result.x * twiddles[512 + j].y));
    u >>= 8;
    j      = u & 255;
    result = ((result.x * twiddles[768 + j].x - result.y * twiddles[768 + j].y),
                                 (result.y * twiddles[768 + j].x + result.x * twiddles[768 + j].y));
    return result;

def chirp_device(grid, threads, unkown, stream, N, M, output, twiddles_large, twl, dir):
    hipBlockDim_x = threads
    for hipBlockIdx_x in range(grid):
        for hipThreadIdx_x in range(threads):
            outfile.write(f"###################### Grid: {hipBlockIdx_x}, Block: {hipThreadIdx_x} ######################\n")
            tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

            val = [0, 0];

            if twl == 1:
                val = TWLstep1(twiddles_large, (tx * tx) % (2 * N));
            elif twl == 2:
                val = TWLstep2(twiddles_large, (tx * tx) % (2 * N));
            elif twl == 3:
                val = TWLstep3(twiddles_large, (tx * tx) % (2 * N));
            elif twl == 4:
                val = TWLstep4(twiddles_large, (tx * tx) % (2 * N));

            val *= dir

            if tx == 0:
                output[tx]     = val;
                output[tx + M] = val;
            elif tx < N:
                output[tx]     = val;
                output[tx + M] = val;
                output[M - tx]     = val;
                output[M - tx + M] = val;
            elif tx <= (M - N):
                output[tx]     = 0;
                output[tx + M] = 0;

def chirp_launch(N, M, B, twiddles_large, twl, dir, rocfft_stream):
    grid = ((M - N) // 64 + 1);
    threads = (64);

    

    outfile.write(f"###################### CHIP ######################\n")
    chirp_device(      grid,
                       threads,
                       0,
                       rocfft_stream,
                       N,
                       M,
                       B,
                       twiddles_large,
                       twl,
                       dir);

    return 0;

def rocfft_internal_chirp():
    N = length[0];
    M = lengthBlue;

    twl = 0;

    if large1D > 256 * 256 * 256 * 256:
        printf("large1D twiddle size too large error");
    elif large1D > 256 * 256 * 256:
        twl = 4;
    elif large1D > 256 * 256:
        twl = 3;
    elif large1D > 256:
        twl = 2;
    else:
        twl = 1;

    dir = direction;

    rocfft_stream = 'null stream';

    chirp_launch(N,
                             M,
                             bufOut[0],
                             twiddles_large,
                             twl,
                             dir,
                             rocfft_stream);


# In[9]:


iDist = 256
oDist = 256
iOffset = 0
oOffset = 0
inStride = [1, iDist]
outStride = [1, oDist]
bufIn = [input_buf, input_buf.sub_array(offset=iDist)]
bufOut = [output_buf, output_buf.sub_array(offset=oDist)]

rocfft_internal_chirp()


# In[ ]:


def mul_device(grid, threads, empty, stream, numof,
                           totalWI,
                           N,
                           M,
                           input,
                           output,
                           dim,
                           lengths,
                           stride_in,
                           stride_out,
                           dir,
                           scheme):
    hipBlockDim_x = threads
    for hipBlockIdx_x in range(grid):
        for hipThreadIdx_x in range(threads):
            outfile.write(f"###################### Grid: {hipBlockIdx_x}, Block: {hipThreadIdx_x} ######################\n")
            tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

            if tx >= totalWI:
                return;

            iOffset = 0;
            oOffset = 0;

            counter_mod = tx // numof;

            #for(i = dim; i > 1; i--)
            for i in range(dim, 1, -1):
                currentLength = 1;
                #for(j = 1; j < i; j++)
                for j in range(1, i):
                    currentLength *= lengths[j];

                iOffset += (counter_mod // currentLength) * stride_in[i];
                oOffset += (counter_mod // currentLength) * stride_out[i];
                counter_mod = counter_mod % currentLength;
            iOffset += counter_mod * stride_in[1];
            oOffset += counter_mod * stride_out[1];

            outfile.write(f"###################### iOffset: {iOffset}, oOffset: {oOffset} ######################\n")

            tx          = tx % numof;
            iIdx = tx * stride_in[0];
            oIdx = tx * stride_out[0];
            if scheme == 0:
                #output += oOffset;
                output = output.sub_array(offset=oOffset)

                out          = output[oIdx];
                output[oIdx] = input[iIdx] * out - input[iIdx] * out;
                output[oIdx] = input[iIdx] * out + input[iIdx] * out;
            elif scheme == 1:
                chirp = output;
                #import pdb; pdb.set_trace()

                #input += iOffset;
                input = input.sub_array(offset=iOffset)

                #output += M;
                #output += oOffset;
                output = output.sub_array(offset=M + oOffset)

                if tx < N:
                    output[oIdx] = input[iIdx] * chirp[tx] + input[iIdx] * chirp[tx];
                    output[oIdx] = -input[iIdx] * chirp[tx] + input[iIdx] * chirp[tx];
                else:
                    output[oIdx] = (0, 0);
            elif scheme == 2:
                chirp = input;

                #input += 2 * M;
                #input += iOffset;
                input = input.sub_array(offset=(2 * M) + iOffset)

                #output += oOffset;
                output = output.sub_array(offset=oOffset)

                MI = 1.0 / M;
                output[oIdx]    = MI * (input[iIdx] * chirp[tx] + input[iIdx] * chirp[tx]);
                output[oIdx]    = MI * (-input[iIdx] * chirp[tx] + input[iIdx] * chirp[tx]);


def rocfft_internal_mul():
    N = length[0];
    M = lengthBlue;

    #// TODO:: fix the local scheme with enum class and pass it
    #//        into kernel as a template parameter
    #scheme = 0; #// fft mul
    #if scheme == CS_KERNEL_PAD_MUL:
    #    scheme = 1; #// pad mul
    #elif scheme == CS_KERNEL_RES_MUL:
    #    scheme = 2; #// res mul

    scheme = 1; #// pad mul

    #cBytes;
    if precision == rocfft_precision_single:
        cBytes = 4 * 2;
    else:
        cBytes = 8 * 2;

    bufIn0  = bufIn[0];
    bufOut0 = bufOut[0];
    bufIn1  = bufIn[1];
    bufOut1 = bufOut[1];

    #// TODO: Not all in/out interleaved/planar combinations support for all 3
    #// schemes until we figure out the buffer offset for planar format.
    #// At least, planar for CS_KERNEL_PAD_MUL input and CS_KERNEL_RES_MUL output
    #// are good enough for current strategy(check TreeNode::ReviseLeafsArrayType).
    #// That is why we add asserts below.

    numof = 0;
    if scheme == 0:
        #bufIn0  = ((char*)bufIn0 + M * cBytes);
        #bufOut0 = ((char*)bufOut0 + 2 * M * cBytes);
        bufIn0  = bufIn0.sub_array(offset=M)
        bufOut0 = bufOut0.sub_array(offset=2 * M)

        numof = M;
    elif scheme == 1:
        #bufOut0 = bufOut0[(M * cBytes) // 4:];
        bufOut0 = bufOut0.sub_array(offset=(M * cBytes) // 4)

        numof = M;
    elif scheme == 2:
        numof = N;

    count = batch;
    #for(i = 1; i < length.size(); i++)
    for i in range(1, len(length)):
        count *= length[i];
    count *= numof;

    dir = direction;

    rocfft_stream = 'null stream'

    grid = ((count - 1) // 64 + 1);
    threads = (64);

    outfile.write(f"###################### MUL ######################\n")
    mul_device(
                       grid,
                       threads,
                       0,
                       rocfft_stream,
                       numof,
                       count,
                       N,
                       M,
                       bufIn0,
                       bufOut0,
                       len(length),
                       length,
                       inStride,
                       outStride,
                       dir,
                       scheme);


# MUL
iDist = 111
oDist = 256
iOffset = 0
oOffset = 0
inStride = [1, iDist]
outStride = [1, oDist]
bufIn = [input_buf, input_buf.sub_array(offset=iOffset)]
bufOut = [output_buf, output_buf.sub_array(offset=oOffset)]

#rocfft_internal_mul()

# FFT_MUL
length = [256]
iDist = 256
oDist = 256
iOffset = 0
oOffset = 0
inStride = [1, iDist]
outStride = [1, oDist]
bufIn = [input_buf, input_buf.sub_array(offset=iOffset)]
bufOut = [output_buf, output_buf.sub_array(offset=oOffset)]

#rocfft_internal_mul()

# RES_MUL
length = [111]
iDist = 256
oDist = 111
iOffset = 0
oOffset = 0
inStride = [1, iDist]
outStride = [1, oDist]
bufIn = [input_buf, input_buf.sub_array(offset=iOffset)]
bufOut = [output_buf, output_buf.sub_array(offset=oOffset)]

rocfft_internal_mul()
