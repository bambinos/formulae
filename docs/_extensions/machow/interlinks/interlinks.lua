local inventory = {} -- sphinx inventories
local autolink       -- set in Meta
local autolink_ignore_token = "qd-no-link"

local function _debug_log(text, debug)
    if debug then
        quarto.log.warning(text)
    end
end

local function read_inv_text(filename)
    local file = io.open(filename, "r")
    if file == nil then
        return nil
    end
    local str = file:read("a")
    file:close()

    local project = str:match("# Project: (%S+)")
    local version = str:match("# Version: (%S+)")
    local data = { project = project, version = version, items = {} }
    local ptn_data =
        "^" ..
        "(.-)%s+" ..
        "([%S:]-):" ..
        "([%S]+)%s+" ..
        "(%-?%d+)%s+" ..
        "(%S*)%s+" ..
        "(.-)\r?$"

    for line in str:gmatch("[^\r\n]+") do
        if not line:match("^#") then
            local name, domain, role, priority, uri, dispName = line:match(ptn_data)
            if name == nil then
                error("Error parsing line: " .. line)
            end
            data.items[#data.items + 1] = {
                name = name,
                domain = domain,
                role = role,
                priority = priority,
                uri = uri,
                dispName = dispName,
            }
        end
    end
    return data
end

local function read_json(filename)
    local file = io.open(filename, "r")
    if file == nil then
        return nil
    end
    local str = file:read("a")
    file:close()
    return quarto.json.decode(str)
end

local function read_inv_text_or_json(base_name)
    local file = io.open(base_name .. ".txt", "r")
    if file then
        io.close(file)
        return read_inv_text(base_name .. ".txt")
    end
    return read_json(base_name .. ".json")
end

local function lookup(search_object, debug)
    local results = {}
    for _, inv in ipairs(inventory) do
        for _, item in ipairs(inv.items) do
            if item.inv_name and item.inv_name ~= search_object.inv_name then
                goto continue
            end
            if item.name ~= search_object.name then
                goto continue
            end
            if search_object.role and item.role ~= search_object.role then
                goto continue
            end
            if search_object.domain and item.domain ~= search_object.domain then
                goto continue
            else
                if search_object.domain or item.domain == "py" then
                    table.insert(results, item)
                end
                goto continue
            end
            ::continue::
        end
    end

    if #results == 1 then
        return results[1]
    end
    if #results > 1 then
        _debug_log("Found multiple matches for object: " .. search_object.name .. ", using the first match.", debug)
        return results[1]
    end
    if #results == 0 then
        _debug_log("Found no matches for object:\n", debug)
        _debug_log(search_object, debug)
    end
    return nil
end

local function mysplit(inputstr, sep)
    if sep == nil then
        sep = "%s"
    end
    local t = {}
    for str in string.gmatch(inputstr, "([^" .. sep .. "]+)") do
        table.insert(t, str)
    end
    return t
end

local function normalize_role(role)
    if role == "func" then
        return "function"
    end
    return role
end

local function copy_replace(original, key, new_value)
    local copy = {}
    for k, v in pairs(original) do
        copy[k] = v
    end
    copy[key] = new_value
    return copy
end

local function contains(list, value)
    for _, v in ipairs(list) do
        if v == value then
            return true
        end
    end
    return false
end

local function flatten_alias_list(list)
    local flat = {}
    for key, sublist in pairs(list) do
        if type(sublist) == "table" then
            for _, subvalue in ipairs(sublist) do
                table.insert(flat, { key, subvalue })
            end
        else
            table.insert(flat, { key, sublist })
        end
    end
    return flat
end

local function prepend_aliases(flat_aliases)
    local new_inv = { project = "aliases", version = "0.0.9999", items = {} }
    for _, name_pair in pairs(flat_aliases) do
        local full = name_pair[1]
        local alias = name_pair[2]
        for _, inv in ipairs(inventory) do
            for _, item in ipairs(inv.items) do
                if string.sub(item.name, 1, string.len(full) + 1) == (full .. ".") then
                    local prefix
                    if not alias or pandoc.utils.stringify(alias) == "" then
                        prefix = ""
                    else
                        prefix = pandoc.utils.stringify(alias) .. "."
                    end
                    local new_name = prefix .. string.sub(item.name, string.len(full) + 2)
                    table.insert(new_inv.items, copy_replace(item, "name", new_name))
                end
            end
        end
    end
    table.insert(inventory, new_inv)
end

local function build_search_object(str, debug)
    local starts_with_colon = str:sub(1, 1) == ":"
    local search = {}
    if starts_with_colon then
        local t = mysplit(str, ":")
        if #t == 2 then
            search.role = normalize_role(t[1])
            search.name = t[2]:match("%%60(.*)%%60")
        elseif #t == 3 then
            search.domain = t[1]
            search.role = normalize_role(t[2])
            search.name = t[3]:match("%%60(.*)%%60")
        elseif #t == 4 then
            search.external = true
            search.inv_name = t[1]:match("external%+(.*)")
            search.domain = t[2]
            search.role = normalize_role(t[3])
            search.name = t[4]:match("%%60(.*)%%60")
        else
            _debug_log("couldn't parse this link: " .. str, debug)
            return {}
        end
    else
        search.name = str:match("%%60(.*)%%60")
    end

    if search.name == nil then
        _debug_log("couldn't parse this link: " .. str, debug)
        return {}
    end
    if search.name:sub(1, 1) == "~" then
        search.shortened = true
        search.name = search.name:sub(2, -1)
    end
    return search
end

local function report_broken_link(link, search_object, replacement)
    return pandoc.Code(pandoc.utils.stringify(link.content))
end

function Link(link)
    if not link.target:match("%%60") then
        return link
    end
    local search = build_search_object(link.target)
    local item = lookup(search)
    local original_text = pandoc.utils.stringify(link.content)
    local replacement = search.name
    if search.shortened then
        local t = mysplit(search.name, ".")
        replacement = t[#t]
    end
    if original_text == "" and replacement ~= nil then
        link.content = pandoc.Code(replacement)
    end
    if item == nil then
        return report_broken_link(link, search)
    end
    link.target = item.uri:gsub("%$$", search.name)
    return link
end

function Code(code)
    if (not autolink) or contains(code.classes, autolink_ignore_token) then
        return code
    end
    local text
    local is_shortened = code.text:sub(1, 2) == "~~"
    local is_short_dot = code.text:sub(1, 3) == "~~."
    local unprefixed = code.text:gsub("^~~%.?", "")
    if unprefixed:match("%(%s*%)") then
        text = unprefixed:gsub("%(%s*%)", "")
    else
        text = unprefixed
    end
    local search = build_search_object("%60" .. text .. "%60")
    local item = lookup(search)
    if item == nil then
        code.text = unprefixed
        return code
    end
    if is_shortened then
        local split = mysplit(unprefixed, ".")
        if #split > 0 then
            local new_name = split[#split]
            if is_short_dot then
                new_name = "." .. new_name
            end
            code.text = new_name
        else
            code.text = unprefixed
        end
    end
    return pandoc.Link(code, item.uri:gsub("%$$", search.name))
end

local function fixup_json(json, prefix)
    for _, item in ipairs(json.items) do
        item.uri = prefix .. item.uri
    end
    table.insert(inventory, json)
end

return {
    {
        Meta = function(meta)
            local json
            local prefix
            local aliases
            if meta.interlinks and meta.interlinks.autolink then
                autolink = true
            else
                autolink = false
            end
            if meta.interlinks and meta.interlinks.aliases then
                aliases = meta.interlinks.aliases
            else
                aliases = {}
            end
            if meta.interlinks and meta.interlinks.sources then
                for k, v in pairs(meta.interlinks.sources) do
                    local base_name = quarto.project.offset .. "/_inv/" .. k .. "_objects"
                    json = read_inv_text_or_json(base_name)
                    prefix = pandoc.utils.stringify(v.url)
                    if json ~= nil then
                        fixup_json(json, prefix)
                    end
                end
            end
            json = read_inv_text_or_json(quarto.project.offset .. "/objects")
            if json ~= nil then
                fixup_json(json, "/")
            end
            prepend_aliases(flatten_alias_list(aliases))
        end,
    },
    {
        Link = Link,
        Code = Code,
    },
}
