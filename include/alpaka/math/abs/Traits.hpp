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
        struct ConceptMathAbs
        {
        };

        //! The math traits.
        namespace traits
        {
            //! The abs trait.
            template<typename T, typename TArg, typename TSfinae = void>
            struct Abs
            {
                ALPAKA_FN_HOST_ACC auto operator()(T const& ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(ctx);
                    // This is an ADL call. If you get a compile error here then your type is not supported by the
                    // backend and we could not find abs(TArg) in the namespace of your type.
                    using std::abs;
                    return abs(arg);
                }
            };
        } // namespace traits

        //! Computes the absolute value.
        //!
        //! \tparam T The type of the object specializing Abs.
        //! \tparam TArg The arg type.
        //! \param abs_ctx The object specializing Abs.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename TArg>
        ALPAKA_FN_HOST_ACC auto abs(T const& abs_ctx, TArg const& arg)
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathAbs, T>;
            return traits::Abs<ImplementationBase, TArg>{}(abs_ctx, arg);
        }
    } // namespace math
} // namespace alpaka
