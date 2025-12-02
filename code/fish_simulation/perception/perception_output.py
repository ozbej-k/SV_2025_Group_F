'''Output formatting for perception module.'''

def build_perception_summary(perception):
    """
    Example output:
    {
      'fish': [{'mu':..., 'A':...}, ...],
      'spots': [...],
      'ATf': total_fish_solid_angle,
      'ATs': total_spot_solid_angle,
      'wall_state': {...}
    }
    """
    ATf = sum(item['A'] for item in perception.get('fish', []))
    ATs = sum(item['A'] for item in perception.get('spots', []))
    out = {
        'fish': perception.get('fish', []),
        'spots': perception.get('spots', []),
        'ATf': ATf,
        'ATs': ATs,
        'wall_state': perception.get('wall_state', {})
    }
    return out
