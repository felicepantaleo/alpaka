/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Unused.hpp>

#include <cmath>

namespace alpaka
{
    namespace math
    {
        struct ConceptMathCbrt
        {
        };

        namespace traits
        {
            //! The cbrt trait.
            template<typename T, typename TArg, typename TSfinae = void>
            struct Cbrt
            {
                ALPAKA_FN_HOST_ACC auto operator()(T const& ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(ctx);
                    // This is an ADL call. If you get a compile error here then your type is not supported by the
                    // backend and we could not find cbrt(TArg) in the namespace of your type.
                    using std::cbrt;
                    return cbrt(arg);
                }
            };
        } // namespace traits

        //! Computes the cbrt.
        //!
        //! \tparam T The type of the object specializing Cbrt.
        //! \tparam TArg The arg type.
        //! \param cbrt_ctx The object specializing Cbrt.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename TArg>
        ALPAKA_FN_HOST_ACC auto cbrt(T const& cbrt_ctx, TArg const& arg)
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathCbrt, T>;
            return traits::Cbrt<ImplementationBase, TArg>{}(cbrt_ctx, arg);
        }
    } // namespace math
} // namespace alpaka
