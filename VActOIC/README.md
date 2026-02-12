# VActOIC
This plugin allows for loading VAct Json (Vson) type files and instantiate entire compositions of objects like meshes, and particles, also includes construction ``SceneComponent`` or ``ActorComponent`` onto object that have these meta data assigned. you can now also create new ``OICProfiles`` from the content browser `richt-click`.

![Static Badge](https://img.shields.io/badge/Important-purple)
_This plugin depends on the ``VActFiles`` plugin, formats compact and binary ar not yet supported_

## Usage
- Create a ``OICProfile`` (Blueprint), reference it to a file
- Drag a ``OICActor`` into the scene, select ``OICComponent`` and assign a created ``OICProfile`` and press update
- Profile instantiation can also be managed in ``OICManagerActor``

![Static Badge](https://img.shields.io/badge/Warning-yellow)
_Update breaks using tracking, sor reloading atm requires you to delete the OICActor and reassign and update again_

## OIC Format V3 (oic.v3, vson.v2)
First `vson` is a extended json version with some extra atomic types, it may even have more in common with `yaml` or `go`. where `vson` allows for
- literal-less property names `My_Property: <vson>`
- var name types `My_Var_Name`
- scientific notation `-1.0e-20`
- float shorthands notations `.2`
- spacial values like `nan`, `true`, `false`, `null`, `inf`
- explicit sign like `+2` or `-3` or `-inf` or `+nan`
- hex numbers `#00000000`
- blob string identifier `&"<binary-data>"`
- reference identifier `@My_Var_Name` or `@"<custom-id>"`
- tuple notation `(<vson> [, <vson>])` e.g. `(1.0, -1.0, 0.42e-7)`

Secondly note that whenever written `"<Tag1|...|TagN>"` in string these are to be replaced with one of the tags `"Tag1"`, Same holds for `<Tag1|...|TagN>` replaces with `Tag`.

Keep in mind that `Meta` entry format is still being worked on. oic value types for `Meta-Properties` specifically `Float#` and `Int#` range from `2 to 6`. 
```go
{
    Type: OIC,
    Version: V3,
    Axis: <UnrealEngine|Unity|Blender|Houdini|TreeJS|BabylonJs>,
    Notes: "some human readable notes",
    Title: "some human readable title",
    Name: OIC_Name_Id,
    Objects: [
        {
            Type: <Mesh|Actor|Particle|Data|Audio|System|Collider>,
            Asset: "Asset<Axis>::Id",
            Meta: 0
        }
    ],
    Instances: [
        (0,0,-1,0,((0.0,0.0,0.0),(0.0,0.0,0.0,1.0),(1.0,1.0,1.0)))
    ],
    Metas: [
        [
            {
                Asset: "Class<Axis>::Id",
                Properties: [
                    {
                        Key: Property_Name_Id,
                        Type: <Name|String|Bool|Float|Float#|Floats|Int|Int#|Ints|Names|Strings|Asset>
                        Value: <<name>|<tuple>|<string>|<num>>
                    }
                ]
            }
        ]
    ]
}
```

instance tuple is composed of `(<id>,<object-id>,<parent-id>,<meta-id>,(<location>,<quaternion>,<scale>))`, `Metas` section is being reworked atm for OIC format `V4`.

## WIP OIC Format V4 (oic.v4, vson.v3)
- vson.v3 will allow tag atomic notation `<Tag>` or flags '<Tag0|...|TagN>` to separate enum types from name types
- vson.v3 will allow name var chaining `Physics.Info.Mass`
- `Object.Name` property is to be an `Axis` independent identifier to resolve assets in other `Axis`
- `Object.CLass` is added to instantiate an asset of a particular class/component.
- `Instance.Index` is added to the type to track particles in instance buffers. or respectively to `Object.Class`
- Packed the meta into a tuple of more general form, allowing `Custom` property type
- meta also allows recursion of its own property map using type `Properties`
- `Metas` will likely be stored in a separate file with same name `L_My_File.oic` with `L_My_File.oicm`

Do not confuse the tag notation below with the actual tag for vson.
```go
{
    Type: OIC,
    Version: V4,
    Axis: <UnrealEngine|Unity|Blender|Houdini|TreeJS|BabylonJs|Unknown>,
    Notes: "some human readable notes",
    Title: "some human readable title",
    Name: OIC_Name_ID,
    Objects: [
        {
            Type: <Mesh|Actor|Particle|Data|Audio|System|Collider>,
            Name: Object_Name_ID
            Class: "Class<Axis>::Id"
            Asset: "Asset<Axis>::Id",
            Meta: 0
        }
    ],
    Instances: [
        (0,0,-1,0,-1,((0.0,0.0,0.0),(0.0,0.0,0.0,1.0),(1.0,1.0,1.0)))
    ],
    Metas: [
        (
            0, 
            "Class<Axis>::Id|Asset<Axis>::Id",
            [
                (
                    <Name|String|Bool|Float|Float#|Floats|Int|Int#|Ints|Names|Strings|Asset|Class|Properties|Custom>,
                    Property_Name_Id,
                    <vson>
                )
            ]
        )
    ]
}
```
```go
type Instances (<id>,<object-id>,<parent-id>,<index>,<meta-id>,(<location>,<quaternion>,<scale>))
type Meta (<id>,<class>,[(<type>, <name>, <vson>)])
```

## WIP VSON V4 (vson.v4)
A major focus on allowing for circular object references.
- struct internal references global `*Version` and relative `*.Asset`
- array/tuple internal references global `*[0]` and relative `*.[0]`
- self and parent chaining `*..[0]` or self `*.`, parent `*..`, scope up chaining `*...`
- chaining of access references `*Objects[0].Asset`
- allow for query reference `*?Objects[0].Name` nearest match global
- allow for query reference `*?.Objects[0].Name` nearest match relative
- allow for query reference `*?<Objects[0].Name` nearest match up
- allow for query reference `*?>Objects[0].Name` nearest match down
- index/tuple access reference `*Metas[.Meta]` to evaluate on any queries
- struct access reference `*NameToValue{.Name}` to evaluate on any queries
- string property reference `*"a struct with spaced string property"`
- number property reference `*0`

A focus on extending some atomic types

- support for named tuple/struct/array/string/blob/reference/tag/hex `Vector(1,2,3)`, `Class"/Path/To/Class`
- support for path atomic type `\Path\With Space\` and relative `.\Path` or `..\Path`
- string mixture in path `\Path\"My string path"\Next`
- regex mixture in path `\Path\/\w+\.*(png|jpg)/`
- string concatenation support `"string0" "string2"`
- regex atomic type `/.*_v\n[0-9a-zA-Z]]/` as `/<regex>/`
- maybe comment block support `/**/`