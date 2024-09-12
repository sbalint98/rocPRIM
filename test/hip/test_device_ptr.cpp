// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "../rocprim/test_utils_device_ptr.hpp"
#include "common_test_header.hpp"

#define DEBUG_CHECK_VEC(vec)                                                    \
    {                                                                           \
        std::cout << "DEBUG:LINE" << __LINE__ << ":CHECK:\t" << #vec << " = ["; \
        auto start = vec.begin();                                               \
        auto end   = vec.end();                                                 \
        while(start != end)                                                     \
        {                                                                       \
            std::cout << *start;                                                \
            if(++start != end)                                                  \
            {                                                                   \
                std::cout << ", ";                                              \
            }                                                                   \
            else                                                                \
            {                                                                   \
                std::cout << "]\n";                                             \
            }                                                                   \
        }                                                                       \
    }                                                                           \
    (void)0

#define DEBUG_CHECK(x)                                                                         \
    {                                                                                          \
        std::cout << "DEBUG:LINE" << __LINE__ << ":CHECK:\t" << #x << " = " << x << std::endl; \
    }                                                                                          \
    (void)0

#define DEBUG_INFO(...)                                                              \
    {                                                                                \
        std::cout << "DEBUG:LINE" << __LINE__ << ":INFO:\t" << #__VA_ARGS__ << "\n"; \
    }                                                                                \
    (void)0

#define DEBUG_RUN(...)                                                              \
    __VA_ARGS__;                                                                    \
    {                                                                               \
        std::cout << "DEBUG:LINE" << __LINE__ << ":RUN:\t" << #__VA_ARGS__ << "\n"; \
    }                                                                               \
    (void)0

TEST(TestDevicePtr, Main)
{
    DEBUG_RUN(std::vector<int> in = {1, 2, 3, 4});
    DEBUG_RUN(test_utils::device_ptr<int> p1(10));
    EXPECT_EQ(p1.size(), 10);

    DEBUG_RUN(p1.store(in));
    DEBUG_RUN(p1.store(in, 4));
    DEBUG_RUN(p1.store(in, 8));
    EXPECT_EQ(p1.size(), 12);

    DEBUG_RUN(auto out1 = p1.load());
    EXPECT_EQ(out1.size(), p1.size());

    DEBUG_RUN(test_utils::device_ptr<int> p2);
    DEBUG_RUN(p2.resize(p1.size()));
    EXPECT_EQ(p2.size(), p1.size());
    DEBUG_RUN(p2.store(p1));

    DEBUG_RUN(auto out2 = p2.load());
    EXPECT_EQ(out2.size(), p2.size());
    EXPECT_EQ(out1, out2);

    EXPECT_EQ(p2.msize(), sizeof(decltype(p2)::value_type) * p2.size());
    EXPECT_EQ(p1.msize(), p1.msize());

    DEBUG_RUN(p2.free_manually());
    EXPECT_EQ(p2.size(), 0);

    DEBUG_RUN(p2.resize(3));
    EXPECT_EQ(p2.size(), 3);

    DEBUG_RUN(p1.resize(10));
    DEBUG_RUN(p2.store(p1));

    DEBUG_RUN(auto out3 = p2.move_cast<void>().load());
    EXPECT_EQ(out3.size(), 10 * sizeof(int));
    EXPECT_EQ(p2.size(), 0);
}