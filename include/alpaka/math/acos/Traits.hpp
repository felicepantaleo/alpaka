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
        struct ConceptMathAcos
        {
        };

        namespace traits
        {
            //! The acos trait.
            template<typename T, typename TArg, typename TSfinae = void>
            struct Acos
            {
                ALPAKA_FN_HOST_ACC auto operator()(T const& ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(ctx);
                    // This is an ADL call. If you get a compile error here then your type is not supported by the
                    // backend and we could not find acos(TArg) in the namespace of your type.
                    using std::acos;
                    return acos(arg);
                }
            };
        } // namespace traits

        //! Computes the principal value of the arc cosine.
        //!
        //! The valid real argument range is [-1.0, 1.0]. For other values
        //! the result may depend on the backend and compilation options, will
        //! likely be NaN.
        //!
        //! \tparam TArg The arg type.
        //! \param acos_ctx The object specializing Acos.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename TArg>
        ALPAKA_FN_HOST_ACC auto acos(T const& acos_ctx, TArg const& arg)
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathAcos, T>;
            return traits::Acos<ImplementationBase, TArg>{}(acos_ctx, arg);
        }
    } // namespace math
} // namespace alpaka
