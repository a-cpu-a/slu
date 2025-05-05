/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <algorithm>
#include <numeric>

#include <slua/parser/State.hpp>
#include <slua/lang/BasicState.hpp>

namespace slua::mlvl
{
    enum class Assoc : uint8_t { LEFT, RIGHT };


    template<bool isLua>
    constexpr uint8_t precedence(parse::BinOpType op) {
        switch (op)
        {
        case parse::BinOpType::EXPONENT: return 90;

        case parse::BinOpType::MODULO: if constexpr (!isLua)return 70;

        case parse::BinOpType::DIVIDE:
        case parse::BinOpType::FLOOR_DIVIDE:
        case parse::BinOpType::MULTIPLY: return 80;


        case parse::BinOpType::SUBTRACT:
        case parse::BinOpType::ADD: return 60;

        case parse::BinOpType::SHIFT_LEFT: 
        case parse::BinOpType::SHIFT_RIGHT: return 50;

        case parse::BinOpType::BITWISE_OR:if constexpr (isLua)return 40;
        case parse::BinOpType::BITWISE_XOR:if constexpr (isLua)return 41;
        case parse::BinOpType::BITWISE_AND: return 42;

        case parse::BinOpType::RANGE_BETWEEN: if constexpr (!isLua)return 30;
            break;

        case parse::BinOpType::CONCATENATE:if constexpr (!isLua) return 20;
            return 55;//Lua, between +- and << >>


        case parse::BinOpType::GREATER_EQUAL:
        case parse::BinOpType::GREATER_THAN:
        case parse::BinOpType::LESS_EQUAL:
        case parse::BinOpType::LESS_THAN: if constexpr(!isLua)return 11;

        case parse::BinOpType::EQUAL:
        case parse::BinOpType::NOT_EQUAL: return 10;


        case parse::BinOpType::LOGICAL_AND: return 1;
        case parse::BinOpType::LOGICAL_OR: return 0;

        case parse::BinOpType::NONE:
            break;
        }
        Slua_panic("Unknown operator, no precedence<slua>() defined");
    }

    template<bool isLua>
    constexpr Assoc associativity(parse::BinOpType op) {
        if constexpr (isLua)
            return op == parse::BinOpType::EXPONENT ? Assoc::RIGHT : Assoc::LEFT;

        switch (op)
        {
        case parse::BinOpType::EXPONENT:
        case parse::BinOpType::SHIFT_LEFT:
        case parse::BinOpType::SHIFT_RIGHT:
            return Assoc::RIGHT;
        default:
            return Assoc::LEFT;
        }
    }

    // Returns the order of operations as indices into `extra`
    // Store parse::BinOpType in `extra[...].first`
    template<bool isLua>
    constexpr std::vector<size_t> multiOpOrder(const auto& m) {
        std::vector<size_t> indices(m.extra.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            auto pa = precedence<isLua>(m.extra[a].first);
            auto pb = precedence<isLua>(m.extra[b].first);
            if (pa != pb) return pa > pb;

            Assoc assoc = associativity<isLua>(m.extra[a].first); // same as b
            if (assoc == Assoc::LEFT)
                return a < b;
            else
                return a > b;
            });

        return indices;
    }
}